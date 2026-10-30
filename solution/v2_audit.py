"""Paired comparison on released periods and public hulls excluded from V2 fitting."""

import argparse, json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from simulation import BASE
from select_and_audit import Ensemble, cluster_interval
from v2_predict import DirectModel
from v2_simulation import DiverseSimulator, FAMILIES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="v2")
    args = parser.parse_args()
    torch.set_num_threads(4)
    old = Ensemble("development", "neural_development")
    old_spec = json.loads((BASE / "reports/frozen_selection.json").read_text())
    new = DirectModel("v2_development")
    sim = DiverseSimulator(["audit_time", "audit_site"], ["audit"])
    results = {}
    frames = []
    for family in FAMILIES + ("held_channel",):
        db, y, meta = sim.generate(1600, 918477, family=family, paired_channels=True)
        components, old_evidence, _ = old.predict_components(db)
        old_prediction = components @ np.asarray(old_spec["weights"])
        prediction, evidence = new.predict(db)
        results[family] = {
            "n": len(y),
            "old_rmse": float(np.sqrt(np.mean((old_prediction - y) ** 2))),
            "v2_rmse": float(np.sqrt(np.mean((prediction - y) ** 2))),
            "v2_bias": float(np.mean(prediction - y)),
            "old_auc": float(roc_auc_score(meta.is_added, old_evidence)),
            "v2_auc": float(roc_auc_score(meta.is_added, evidence)),
            "v2_rmse_cluster_interval": cluster_interval(y, prediction, meta),
        }
        meta["old_prediction"] = old_prediction
        meta["v2_prediction"] = prediction
        meta["evidence"] = evidence
        frames.append(meta)
        print(family, results[family], flush=True)
    pd.concat(frames, ignore_index=True).to_csv(
        BASE / "reports" / (args.label + "_audit_predictions.csv.gz"), index=False
    )
    (BASE / "reports" / (args.label + "_audit_metrics.json")).write_text(
        json.dumps(results, indent=2)
    )


if __name__ == "__main__":
    main()
