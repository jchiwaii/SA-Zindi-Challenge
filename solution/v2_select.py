"""Combine manually specified candidates using validation predictions only."""

import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from simulation import BASE
from v2_train import REGIME_WEIGHTS


def main():
    tree = BASE / "runs/v2_development"
    a = pd.read_csv(tree / "validation_predictions.csv.gz")
    b = pd.read_csv(BASE / "runs/v2_neural_development/validation_predictions.csv.gz")
    assert a.window_id.equals(b.window_id) and a.reference_member.equals(b.reference_member)
    np.testing.assert_allclose(a.share, b.share, atol=1e-8)
    original = json.loads((tree / "selection.json").read_text())
    p = np.column_stack(
        [a.full, a.temporal, np.full(len(a), original["target_mean"]), b.prediction]
    )
    weight = a.strength.map(REGIME_WEIGHTS).to_numpy()
    weight /= weight.sum()
    y = a.share.to_numpy()
    objective = lambda beta: float(np.sum(weight * (p @ beta - y) ** 2))
    result = minimize(
        objective,
        [0.4, 0.1, 0, 0.5],
        bounds=[(0, 1)] * 4,
        method="SLSQP",
        constraints=[{"type": "eq", "fun": lambda beta: beta.sum() - 1}],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    assert result.success
    beta = np.maximum(result.x, 0)
    beta /= beta.sum()
    auc = {
        "tree": float(roc_auc_score(a.is_added, a.evidence)),
        "neural": float(roc_auc_score(a.is_added, b.evidence)),
    }
    spec = {
        **original,
        "weights": beta.tolist(),
        "components": ["full", "temporal", "training_mean", "neural"],
        "neural_run": "v2_neural_development",
        "evidence_source": max(auc, key=auc.get),
        "validation_auc": auc,
        "weighted_validation_rmse": float(np.sqrt(objective(beta))),
        "component_validation_rmse": {
            name: float(np.sqrt(np.sum(weight * (p[:, i] - y) ** 2)))
            for i, name in enumerate(["full", "temporal", "training_mean", "neural"])
        },
        "selection_uses_audit": False,
    }
    (tree / "combined_selection.json").write_text(json.dumps(spec, indent=2))
    print(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
