"""Compact spectral-temporal network with an energy-weighted decomposition head."""

import argparse, json, resource, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from acoustic_model import features, energy
from simulation import Simulator, BASE
from train import SCENARIOS, SCENARIO_WEIGHTS, metrics, save_json


class AcousticNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(256, 128),
            nn.SiLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 48, 3, padding=2, dilation=2),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.Conv2d(48, 64, 3, padding=3, dilation=3),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((3, 5))
        self.head = nn.Sequential(
            nn.Linear(128 + 64 * 15, 256),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )
        self.cell_head = nn.Sequential(nn.Conv2d(64, 16, 1), nn.SiLU(), nn.Conv2d(16, 1, 1))

    def forward(self, raw, feat, weights):
        hidden = self.conv(raw)
        logits = self.head(torch.cat([self.feature_net(feat), self.pool(hidden).flatten(1)], dim=1))
        cell = torch.sigmoid(self.cell_head(hidden)).squeeze(1)
        physical = (cell * weights).sum((1, 2))
        global_share = torch.sigmoid(logits[:, 0])
        return 0.5 * (physical + global_share), logits[:, 1], cell, physical, global_share


def raw_inputs(db):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        avg = np.nanmean(db, (1, 2), keepdims=True)
        band = np.nanmean(db, 1, keepdims=True)
        raw = np.stack(
            [(db - 80) / 35, (db - avg) / 20, (db - band) / 10, np.isfinite(db).astype(np.float32)],
            1,
        )
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    e, _ = energy(db)
    weights = e / (e.sum((1, 2), keepdims=True) + 1e-25)
    return raw, weights


def neural_predict(model, db, feat, normalizer, device="cpu", batch_size=256):
    result = []
    evidence = []
    physical = []
    global_share = []
    model.eval()
    for i in range(0, len(db), batch_size):
        raw, weight = raw_inputs(db[i : i + batch_size])
        f = np.nan_to_num(
            (feat[i : i + batch_size] - normalizer["mean"]) / normalizer["std"], nan=0.0
        )
        with torch.no_grad():
            outputs = model(
                torch.as_tensor(raw, device=device),
                torch.as_tensor(np.clip(f, -10, 10), device=device),
                torch.as_tensor(weight, device=device),
            )
        result.append(outputs[0].cpu().numpy())
        evidence.append(torch.sigmoid(outputs[1]).cpu().numpy())
        physical.append(outputs[3].cpu().numpy())
        global_share.append(outputs[4].cpu().numpy())
    return (
        np.concatenate(result),
        np.concatenate(evidence),
        np.concatenate(physical),
        np.concatenate(global_share),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="neural_development")
    p.add_argument("--templates", default="development")
    p.add_argument("--n-train", type=int, default=60000)
    p.add_argument("--n-val", type=int, default=1500)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--schedule-epochs", type=int)
    p.add_argument("--final", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--prototype", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(5)
    torch.manual_seed(9173)
    rng = np.random.default_rng(9173)
    device = (
        ("mps" if torch.backends.mps.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    out = BASE / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    templates = np.load(BASE / "runs" / args.templates / "templates.npy")
    np.save(out / "templates.npy", templates)
    sim = Simulator(
        (
            ["train"]
            if not args.final
            else ["train", "val_time", "audit_time", "val_site", "audit_site"]
        ),
        ["train"] if not args.final else ["train", "val", "audit"],
    )
    if not args.prototype:
        assert (BASE / "cache/reference_manifest.csv").exists()
    xs = []
    ys = []
    fs = []
    cs = []
    ms = []
    for i in range(0, args.n_train, 6000):
        x, y, m, c = sim.generate(min(6000, args.n_train - i), 930000 + i, return_cells=True)
        f, names = features(x, templates)
        xs.append(x)
        ys.append(y)
        fs.append(f)
        cs.append(c)
        ms.append(m)
        print("Neural training features", i + len(y), flush=True)
    db = np.concatenate(xs)
    target = np.concatenate(ys)
    feature = np.concatenate(fs)
    cells = np.concatenate(cs)
    meta = pd.concat(ms, ignore_index=True)
    del xs, ys, fs, cs, ms
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        normalizer = {
            "mean": np.nanmean(feature, 0),
            "std": np.maximum(np.nanstd(feature, 0), 1e-4),
        }
    normalizer = {
        k: np.nan_to_num(v, nan=0 if k == "mean" else 1).astype(np.float32)
        for k, v in normalizer.items()
    }
    np.savez(out / "normalizer.npz", **normalizer)
    feature = np.clip(
        np.nan_to_num((feature - normalizer["mean"]) / normalizer["std"], nan=0), -10, 10
    )
    meta.to_csv(out / "training_examples.csv.gz", index=False)
    save_json(
        out / "config.json",
        {
            "args": vars(args),
            "seed": 9173,
            "device": device,
            "feature_count": feature.shape[1],
            "training_source_clips": len(sim.refs),
            "test_inputs_read": False,
        },
    )
    validation = {}
    if not args.final:
        val = Simulator(["val_time", "val_site"], ["val"])
        for j, s in enumerate(SCENARIOS):
            x, y, m = val.generate(
                args.n_val, 800000 + j * 10000, scenario=s, augment_background=False
            )
            f, _ = features(x, templates)
            validation[s] = (x, f, y, m)
    model = AcousticNet(feature.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.schedule_epochs or args.epochs, eta_min=3e-5
    )
    best = float("inf")
    stale = 0
    log = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for ix in np.array_split(rng.permutation(len(db)), int(np.ceil(len(db) / 256))):
            raw, weights = raw_inputs(db[ix])
            outp = model(
                torch.as_tensor(raw, device=device),
                torch.as_tensor(feature[ix], device=device),
                torch.as_tensor(weights, device=device),
            )
            y = torch.as_tensor(target[ix], device=device)
            truth = torch.as_tensor(cells[ix], device=device)
            # All heads learn supervised additions. Cell loss emphasizes energetically relevant cells.
            loss = (
                ((outp[0] - y) ** 2).mean()
                + 0.25 * ((outp[3] - y) ** 2).mean()
                + 0.25 * ((outp[4] - y) ** 2).mean()
            )
            loss += (
                0.03
                * (((outp[2] - truth) ** 2) * torch.as_tensor(weights, device=device))
                .sum((1, 2))
                .mean()
            )
            loss += 0.005 * nn.functional.binary_cross_entropy_with_logits(
                outp[1],
                torch.as_tensor(meta.is_added.to_numpy()[ix].astype(np.float32), device=device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if validation:
            scores = []
            for s in ["standard", "weak", "channel_shift"]:
                x, f, y, m = validation[s]
                pred, *_ = neural_predict(model, x, f, normalizer, device)
                scores.append(float(np.mean((pred - y) ** 2)))
            score = float(np.sqrt(0.6 * scores[0] + 0.25 * scores[1] + 0.15 * scores[2]))
        else:
            score = float(np.mean(losses))
        log.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                ("selection_rmse" if validation else "training_objective"): score,
                "elapsed_seconds": time.time() - started,
            }
        )
        print(log[-1], flush=True)
        if args.final or score < best:
            best = min(best, score)
            stale = 0
            torch.save(
                {
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "n_features": feature.shape[1],
                    "epoch": epoch + 1,
                },
                out / "model.pt",
            )
        else:
            stale += 1
        save_json(out / "learning_curve.json", log)
        if stale >= 12:
            break
    checkpoint = torch.load(out / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    scores = {}
    for s, (x, f, y, m) in validation.items():
        pred, evidence, physical, global_share = neural_predict(model, x, f, normalizer, device)
        scores[s] = metrics(y, pred, m)
        frame = m.copy()
        frame["neural"] = pred
        frame["neural_evidence"] = evidence
        frame["neural_physical"] = physical
        frame["neural_global"] = global_share
        frame.to_csv(out / f"validation_predictions_{s}.csv.gz", index=False)
    if scores:
        scores["weighted_rmse"] = sum(SCENARIO_WEIGHTS[s] * scores[s]["rmse"] for s in SCENARIOS)
    save_json(out / "validation_metrics.json", scores)
    save_json(
        out / "training_summary.json",
        {
            "wall_seconds": time.time() - started,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "epochs": len(log),
            "chosen_epoch": checkpoint["epoch"],
            "device": device,
            "parameters": sum(p.numel() for p in model.parameters()),
            "n_train": len(db),
        },
    )
    print("Neural complete", scores.get("weighted_rmse"), flush=True)


if __name__ == "__main__":
    main()
