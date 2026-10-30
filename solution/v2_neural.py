"""Fit the existing physical-head network on the broader V2 generator."""

import argparse, json, resource, time, warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from neural import AcousticNet, raw_inputs, neural_predict
from simulation import BASE
from v2_models import build_features
from v2_simulation import DiverseSimulator, FAMILIES
from v2_train import REGIMES, REGIME_WEIGHTS, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--n-train", type=int, default=100000)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(71391)
    rng = np.random.default_rng(71391)
    start = time.time()
    out = BASE / "runs" / ("v2_neural_final" if args.final else "v2_neural_development")
    out.mkdir(parents=True, exist_ok=True)
    epochs = (
        json.loads((BASE / "runs/v2_neural_development/training_summary.json").read_text())[
            "chosen_epoch"
        ]
        if args.final
        else 50
    )
    sim = DiverseSimulator(
        ["train", "val_time", "audit_time", "val_site", "audit_site"] if args.final else ["train"],
        ["train", "val", "audit"] if args.final else ["train"],
    )
    raw, target, feature, cells, metadata = [], [], [], [], []
    for i in range(0, args.n_train, 5000):
        x, y, meta, c = sim.generate(min(5000, args.n_train - i), 743000 + i, return_cells=True)
        f, names, _ = build_features(x)
        raw.append(x)
        target.append(y)
        feature.append(f)
        cells.append(c)
        metadata.append(meta)
        print("Prepared neural examples", i + len(y), flush=True)
    db, y, f, c = map(np.concatenate, [raw, target, feature, cells])
    meta = pd.concat(metadata, ignore_index=True)
    del raw, target, feature, cells, metadata
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        normalizer = {"mean": np.nanmean(f, axis=0), "std": np.maximum(np.nanstd(f, axis=0), 1e-4)}
    normalizer = {
        k: np.nan_to_num(v, nan=0 if k == "mean" else 1).astype(np.float32)
        for k, v in normalizer.items()
    }
    np.savez(out / "normalizer.npz", **normalizer)
    f = np.clip(np.nan_to_num((f - normalizer["mean"]) / normalizer["std"], nan=0), -10, 10)
    meta.to_csv(out / "training_examples.csv.gz", index=False)
    vx, vf, vy, vm = [], [], [], []
    if not args.final:
        val = DiverseSimulator(["val_time", "val_site"], ["val"])
        for j, family in enumerate(FAMILIES + ("held_channel",)):
            for k, regime in enumerate(REGIMES):
                xx, yy, mm = val.generate(600, 570000 + j * 100 + k, family, regime)
                ff, _, _ = build_features(xx)
                vx.append(xx)
                vf.append(ff)
                vy.append(yy)
                vm.append(mm)
        vx, vf, vy = map(np.concatenate, [vx, vf, vy])
        vm = pd.concat(vm, ignore_index=True)
        vw = vm.strength.map(REGIME_WEIGHTS).to_numpy()
        vw /= vw.sum()
    model = AcousticNet(f.shape[1]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, eta_min=3e-5)
    best, stale, chosen = float("inf"), 0, 0
    curve = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for ix in np.array_split(rng.permutation(len(y)), int(np.ceil(len(y) / 256))):
            rr, ww = raw_inputs(db[ix])
            weights = torch.as_tensor(ww, device=args.device)
            output = model(
                torch.as_tensor(rr, device=args.device),
                torch.as_tensor(f[ix], device=args.device),
                weights,
            )
            yy = torch.as_tensor(y[ix], device=args.device)
            loss = (
                ((output[0] - yy) ** 2).mean()
                + 0.25 * ((output[3] - yy) ** 2).mean()
                + 0.25 * ((output[4] - yy) ** 2).mean()
            )
            loss += (
                0.03
                * (((output[2] - torch.as_tensor(c[ix], device=args.device)) ** 2) * weights)
                .sum((1, 2))
                .mean()
            )
            loss += 0.005 * nn.functional.binary_cross_entropy_with_logits(
                output[1],
                torch.as_tensor(
                    meta.is_added.to_numpy()[ix].astype(np.float32), device=args.device
                ),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        if args.final:
            metric = float(np.mean(losses))
        else:
            prediction, evidence, _, _ = neural_predict(model, vx, vf, normalizer, args.device)
            metric = float(np.sqrt(np.sum(vw * (prediction - vy) ** 2)))
        record = {
            "epoch": epoch + 1,
            "training_loss": float(np.mean(losses)),
            "training_objective" if args.final else "validation_rmse": metric,
            "seconds": time.time() - start,
        }
        curve.append(record)
        print(record, flush=True)
        if args.final or metric < best:
            best, stale, chosen = min(best, metric), 0, epoch + 1
            torch.save(
                {
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "n_features": f.shape[1],
                    "epoch": epoch + 1,
                },
                out / "model.pt",
            )
            if not args.final:
                frame = vm.copy()
                frame["prediction"], frame["evidence"] = prediction, evidence
                frame.to_csv(out / "validation_predictions.csv.gz", index=False)
        else:
            stale += 1
        write_json(out / "learning_curve.json", curve)
        if stale >= 9:
            break
    write_json(
        out / "training_summary.json",
        {
            "n_train": len(y),
            "seed": 71391,
            "device": args.device,
            "chosen_epoch": chosen,
            "epochs_run": len(curve),
            "cosine_schedule_epochs": 50,
            "wall_seconds": time.time() - start,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "is_final": args.final,
            "test_used": False,
        },
    )
    print("Finished neural", out.name, flush=True)


if __name__ == "__main__":
    main()
