"""Evaluate the original frozen model under training-only channel interventions."""

import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from simulation import BASE
from select_and_audit import Ensemble
from v2_simulation import DiverseSimulator, FAMILIES


def main():
    torch.set_num_threads(4)
    model = Ensemble("development", "neural_development")
    spec = json.loads((BASE / "reports/frozen_selection.json").read_text())
    sim = DiverseSimulator(["val_time", "val_site"], ["val"])
    records = {}
    for family in FAMILIES + ("held_channel",):
        x, y, meta = sim.generate(1600, 881700, family=family, paired_channels=True)
        components, evidence, _ = model.predict_components(x)
        prediction = components @ np.asarray(spec["weights"])
        raw = components[:, :4] @ np.asarray(spec["weights"])[4:]
        records[family] = {
            "n": len(y),
            "rmse_original": float(np.sqrt(np.mean((prediction - y) ** 2))),
            "rmse_without_gate": float(np.sqrt(np.mean((raw - y) ** 2))),
            "evidence_auc": float(roc_auc_score(meta.is_added, evidence)),
            "bias": float(np.mean(prediction - y)),
        }
        print(family, records[family], flush=True)
    (BASE / "reports/v2_failure_diagnosis.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
