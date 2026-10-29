"""Fit manually specified robust regressors; test inputs are never opened."""

import argparse, json, resource, time
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from simulation import BASE
from v2_models import build_features
from v2_simulation import DiverseSimulator, FAMILIES

REGIMES = ("weak", "broad", "strong")
REGIME_WEIGHTS = {"weak": 0.4, "broad": 0.4, "strong": 0.2}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2))


def evaluate(y, prediction, evidence, meta):
    result = {}
    for (family, strength), group in meta.groupby(["family", "strength"]):
        ix = group.index.to_numpy()
        result[f"{family}/{strength}"] = {
            "n": len(ix),
            "rmse": float(np.sqrt(np.mean((prediction[ix] - y[ix]) ** 2))),
            "bias": float(np.mean(prediction[ix] - y[ix])),
            "auc": float(roc_auc_score(group.is_added, evidence[ix])),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--n-train", type=int, default=120000)
    args = parser.parse_args()
    start = time.time()
    name = "v2_final" if args.final else "v2_development"
    out = BASE / "runs" / name
    out.mkdir(exist_ok=True, parents=True)
    prior_spec = (
        json.loads((BASE / "runs/v2_development/selection.json").read_text())
        if args.final
        else None
    )
    sim = DiverseSimulator(
        ["train", "val_time", "audit_time", "val_site", "audit_site"] if args.final else ["train"],
        ["train", "val", "audit"] if args.final else ["train"],
    )
    db, y, meta = sim.generate(args.n_train, 44019)
    meta.to_csv(out / "training_examples.csv.gz", index=False)
    x, names, invariant = build_features(db)
    del db
    write_json(out / "features.json", {"names": names, "invariant_columns": invariant.tolist()})
    print("Training examples and features ready", x.shape, flush=True)
    if not args.final:
        val_sim = DiverseSimulator(["val_time", "val_site"], ["val"])
        vx, vy, vm = [], [], []
        for j, family in enumerate(FAMILIES + ("held_channel",)):
            for k, regime in enumerate(REGIMES):
                d, target, details = val_sim.generate(600, 570000 + j * 100 + k, family, regime)
                f, _, _ = build_features(d)
                vx.append(f)
                vy.append(target)
                vm.append(details)
        val_x, val_y, val_meta = (
            np.concatenate(vx),
            np.concatenate(vy),
            pd.concat(vm, ignore_index=True),
        )
        del vx, vy, vm
    counts = {}
    predictions = []
    for label, columns, leaves in [
        ("full", np.arange(x.shape[1]), 40),
        ("temporal", invariant, 31),
    ]:
        iterations = prior_spec["iterations"][label] if args.final else 2200
        model = lgb.LGBMRegressor(
            n_estimators=iterations,
            learning_rate=0.035,
            num_leaves=leaves,
            max_bin=127,
            min_child_samples=140,
            reg_lambda=4.0,
            colsample_bytree=0.85,
            subsample=0.85,
            subsample_freq=1,
            random_state=8371,
            n_jobs=6,
            verbosity=-1,
        )
        options = (
            {}
            if args.final
            else {
                "eval_set": [(val_x[:, columns], val_y)],
                "callbacks": [lgb.early_stopping(120, verbose=False), lgb.log_evaluation(300)],
            }
        )
        model.fit(x[:, columns], y, **options)
        joblib.dump(model, out / (label + ".joblib"))
        counts[label] = model.n_estimators_
        print("Saved", label, counts[label], flush=True)
        if not args.final:
            predictions.append(np.clip(model.predict(val_x[:, columns]), 0, 1))
    classifier = lgb.LGBMClassifier(
        n_estimators=prior_spec["iterations"]["evidence"] if args.final else 1200,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=160,
        reg_lambda=5,
        max_bin=127,
        colsample_bytree=0.85,
        random_state=9371,
        n_jobs=6,
        verbosity=-1,
    )
    options = (
        {}
        if args.final
        else {
            "eval_set": [(val_x, val_meta.is_added)],
            "eval_metric": "auc",
            "callbacks": [lgb.early_stopping(100, first_metric_only=True, verbose=False)],
        }
    )
    classifier.fit(x, meta.is_added, **options)
    joblib.dump(classifier, out / "evidence.joblib")
    counts["evidence"] = classifier.n_estimators_
    if args.final:
        spec = dict(prior_spec)
        spec["target_mean"] = float(y.mean())
        spec["addition_rate"] = float(meta.is_added.mean())
    else:
        prediction = np.column_stack(predictions + [np.full(len(val_y), y.mean())])
        evidence = classifier.predict_proba(val_x)[:, 1]
        weights = val_meta.strength.map(REGIME_WEIGHTS).to_numpy()
        weights /= weights.sum()
        objective = lambda blend: float(np.sum(weights * (prediction @ blend - val_y) ** 2))
        fit = minimize(
            objective,
            [0.5, 0.5, 0],
            method="SLSQP",
            bounds=[(0, 1)] * 3,
            constraints=[{"type": "eq", "fun": lambda b: b.sum() - 1}],
            options={"ftol": 1e-12, "maxiter": 500},
        )
        assert fit.success
        blend = np.maximum(fit.x, 0)
        blend /= blend.sum()
        final_prediction = prediction @ blend
        spec = {
            "weights": blend.tolist(),
            "components": ["full", "temporal", "training_mean"],
            "iterations": counts,
            "target_mean": float(y.mean()),
            "addition_rate": float(meta.is_added.mean()),
            "test_used_for_fitting": False,
            "weighted_validation_rmse": float(np.sqrt(objective(blend))),
            "evidence_gating": False,
        }
        val_meta["prediction"] = final_prediction
        val_meta["full"] = prediction[:, 0]
        val_meta["temporal"] = prediction[:, 1]
        val_meta["evidence"] = evidence
        val_meta.to_csv(out / "validation_predictions.csv.gz", index=False)
        write_json(
            out / "validation_metrics.json", evaluate(val_y, final_prediction, evidence, val_meta)
        )
    write_json(out / "selection.json", spec)
    write_json(
        out / "training_summary.json",
        {
            "n_train": len(y),
            "seed": 44019,
            "wall_seconds": time.time() - start,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "iterations": counts,
            "is_final": args.final,
        },
    )
    print("Finished", name, spec, flush=True)


if __name__ == "__main__":
    main()
