"""Freeze a validation-selected blend, then evaluate untouched audit partitions."""

import argparse, json, resource, time
import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from acoustic_model import features
from simulation import Simulator, BASE
from neural import AcousticNet, neural_predict
from train import SCENARIOS, SCENARIO_WEIGHTS, metrics, save_json

BASE_COMPONENTS = ["lightgbm", "catboost", "conservative", "neural"]
COMPONENTS = BASE_COMPONENTS + [name + "_evidence_weighted" for name in BASE_COMPONENTS]


class Ensemble:
    def __init__(self, tree_dir, neural_dir, device="cpu"):
        self.tree_dir = BASE / "runs" / tree_dir
        self.neural_dir = BASE / "runs" / neural_dir
        self.templates = np.load(self.tree_dir / "templates.npy")
        self.models = {
            k: joblib.load(self.tree_dir / (k + ".joblib"))
            for k in ["lightgbm", "conservative", "evidence"]
        }
        self.cat = CatBoostRegressor()
        self.cat.load_model(str(self.tree_dir / "catboost.cbm"))
        ck = torch.load(self.neural_dir / "model.pt", map_location="cpu", weights_only=True)
        self.neural = AcousticNet(ck["n_features"]).to(device)
        self.neural.load_state_dict(ck["state_dict"])
        self.device = device
        self.normalizer = dict(np.load(self.neural_dir / "normalizer.npz"))
        np.testing.assert_array_equal(self.templates, np.load(self.neural_dir / "templates.npy"))

    def predict_components(self, db):
        x, names = features(db, self.templates)
        a = np.clip(self.models["lightgbm"].predict(x), 0, 1)
        b = np.clip(self.cat.predict(np.nan_to_num(x, nan=-9999.0)), 0, 1)
        c = np.clip(self.models["conservative"].predict(x), 0, 1)
        d, nev, _, _ = neural_predict(self.neural, db, x, self.normalizer, self.device)
        ev = self.models["evidence"].predict_proba(x)[:, 1]
        base = np.column_stack([a, b, c, d])
        return np.column_stack([base, base * ev[:, None]]), ev, nev


def cluster_interval(y, p, meta, repeats=500):
    background_meta = pd.read_csv(BASE / "cache/background_windows.csv")
    clock = (
        background_meta.loc[meta.background_index.to_numpy(), "start_export_clock"]
        .str.slice(0, 10)
        .to_numpy(dtype=str)
    )
    site = meta.site.astype(str).to_numpy(dtype=str)
    bg_groups = np.unique(np.char.add(site, clock), return_inverse=True)[1]
    ref_groups = np.unique(meta.reference_hull, return_inverse=True)[1]
    rng = np.random.default_rng(81273)
    values = []
    err = (y - p) ** 2
    for _ in range(repeats):
        a = rng.poisson(1, bg_groups.max() + 1)
        b = rng.poisson(1, ref_groups.max() + 1)
        w = a[bg_groups] * b[ref_groups]
        if w.sum():
            values.append(np.sqrt(np.sum(w * err) / w.sum()))
    return np.quantile(values, [0.025, 0.5, 0.975]).tolist()


def select(tree_dir, neural_dir, out):
    pieces = []
    targets = []
    weights = []
    for scenario in SCENARIOS:
        a = pd.read_csv(BASE / "runs" / tree_dir / f"validation_predictions_{scenario}.csv.gz")
        b = pd.read_csv(BASE / "runs" / neural_dir / f"validation_predictions_{scenario}.csv.gz")
        assert a.window_id.tolist() == b.window_id.tolist()
        assert a.reference_member.tolist() == b.reference_member.tolist()
        np.testing.assert_allclose(a.share, b.share, atol=1e-8)
        evidence = pd.read_csv(
            BASE / "runs" / tree_dir / f"validation_evidence_{scenario}.csv.gz"
        ).evidence.to_numpy()
        base = np.column_stack([a[c] for c in BASE_COMPONENTS[:-1]] + [b.neural])
        pieces.append(np.column_stack([base, base * evidence[:, None]]))
        targets.append(a.share.to_numpy())
        weights.append(np.full(len(a), SCENARIO_WEIGHTS[scenario] / len(a)))
    pred = np.concatenate(pieces)
    y = np.concatenate(targets)
    w = np.concatenate(weights)
    objective = lambda beta: float(np.sum(w * (pred @ beta - y) ** 2))
    result = minimize(
        objective,
        np.full(len(COMPONENTS), 1 / len(COMPONENTS)),
        bounds=[(0, 1)] * len(COMPONENTS),
        constraints=[{"type": "eq", "fun": lambda beta: beta.sum() - 1}],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 200},
    )
    assert result.success
    beta = np.maximum(result.x, 0)
    beta /= beta.sum()
    alternative = np.zeros(len(COMPONENTS))
    alternative[COMPONENTS.index("conservative_evidence_weighted")] = 1
    specification = {
        "tree_run": tree_dir,
        "neural_run": neural_dir,
        "component_order": COMPONENTS,
        "weights": beta.tolist(),
        "conservative_weights": alternative.tolist(),
        "evidence": "lightgbm_classifier",
        "selection_objective": "Weighted validation MSE over six predeclared scenarios",
        "scenario_weights": SCENARIO_WEIGHTS,
        "validation_rmse": float(np.sqrt(objective(beta))),
        "test_inputs_read": False,
        "audit_used_for_selection": False,
    }
    save_json(out / "frozen_selection.json", specification)
    pd.DataFrame(
        {
            "component": COMPONENTS,
            "weight": beta,
            "validation_weighted_rmse": [
                float(np.sqrt(np.sum(w * (pred[:, i] - y) ** 2))) for i in range(len(COMPONENTS))
            ],
        }
    ).to_csv(out / "ensemble_weights.csv", index=False)
    print("Frozen selection:", specification, flush=True)
    return specification


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tree", default="development")
    p.add_argument("--neural", default="neural_development")
    p.add_argument("--n-audit", type=int, default=2500)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(5)
    out = BASE / "reports"
    started = time.time()
    specification = select(args.tree, args.neural, out)
    model = Ensemble(args.tree, args.neural, args.device)
    sim = Simulator(["audit_time", "audit_site"], ["audit"])
    reports = {}
    frames = []
    for j, scenario in enumerate(SCENARIOS):
        x, y, meta = sim.generate(
            args.n_audit, 1700000 + j * 10000, scenario=scenario, augment_background=False
        )
        components, ev, nev = model.predict_components(x)
        pred = components @ np.asarray(specification["weights"])
        conservative = components @ np.asarray(specification["conservative_weights"])
        reports[scenario] = {
            "primary": metrics(y, pred, meta),
            "conservative": metrics(y, conservative, meta),
            "evidence_auc": float(roc_auc_score(meta.is_added, ev)),
            "rmse_cluster_bootstrap_95_interval": cluster_interval(y, pred, meta),
        }
        for i, c in enumerate(COMPONENTS):
            meta[c] = components[:, i]
        meta["prediction"] = pred
        meta["evidence"] = ev
        frames.append(meta)
        if scenario == "standard":
            np.savez_compressed(
                out / "audit_examples.npz", db=x[:240], target=y[:240], prediction=pred[:240]
            )
            controls = {}
            rng = np.random.default_rng(2341)
            for name, changed in [
                ("global_gain_plus6", x[:1200] + 6),
                ("time_reversal", x[:1200, ::-1].copy()),
                ("time_shuffle", x[:1200, rng.permutation(15)].copy()),
                ("frequency_reversal", x[:1200, :, ::-1].copy()),
                ("quantize_to_0p1_db", np.round(x[:1200] * 10) / 10),
                ("quantize_to_0p5_db", np.round(x[:1200] * 2) / 2),
            ]:
                cp, _, _ = model.predict_components(changed)
                pp = cp @ np.asarray(specification["weights"])
                controls[name] = {
                    "rmse": float(np.sqrt(np.mean((pp - y[:1200]) ** 2))),
                    "mean_absolute_prediction_change": float(np.mean(abs(pp - pred[:1200]))),
                }
            reports[scenario]["sensitivity_controls"] = controls
        print(
            "Audit",
            scenario,
            "RMSE",
            reports[scenario]["primary"]["rmse"],
            "AUC",
            reports[scenario]["evidence_auc"],
            flush=True,
        )
    audit = pd.concat(frames, ignore_index=True)
    audit.to_csv(out / "audit_predictions.csv.gz", index=False)
    reports["weighted_rmse"] = sum(
        SCENARIO_WEIGHTS[s] * reports[s]["primary"]["rmse"] for s in SCENARIOS
    )
    reports["n_examples"] = len(audit)
    reports["source_hulls"] = int(audit.reference_hull.nunique())
    reports["background_windows"] = int(audit.window_id.nunique())
    reports["wall_seconds"] = time.time() - started
    save_json(out / "audit_metrics.json", reports)
    print("Audit completed without using competition test inputs.", flush=True)


if __name__ == "__main__":
    main()
