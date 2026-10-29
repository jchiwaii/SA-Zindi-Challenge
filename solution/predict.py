"""Inference only: fixed transformations, saved weights, and exact ID validation."""

import argparse, hashlib, json, resource, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from audit_data import COLS
from select_and_audit import Ensemble
from simulation import BASE


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", type=Path, default=BASE.parent / "Test.csv")
    p.add_argument("--sample", type=Path, default=BASE.parent / "sample_submission.csv")
    p.add_argument("--output", type=Path, default=BASE.parent / "submission.csv")
    p.add_argument("--alternative", type=Path, default=BASE.parent / "submission_conservative.csv")
    p.add_argument("--tree", default="final")
    p.add_argument("--neural", default="neural_final")
    p.add_argument("--selection", type=Path, default=BASE / "reports/frozen_selection.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--expected-rows", type=int, default=1500)
    args = p.parse_args()
    started = time.time()
    torch.set_num_threads(5)
    spec = json.loads(args.selection.read_text())
    assert spec["test_inputs_read"] is False and spec["audit_used_for_selection"] is False
    # Validate the frozen model list before opening competition acoustic inputs.
    model = Ensemble(args.tree, args.neural, args.device)
    summary = json.loads((BASE / "runs" / args.tree / "training_summary.json").read_text())
    feature_cols = [f"m{m:02d}_{c}" for m in range(1, 16) for c in COLS]
    test = pd.read_csv(args.test, dtype={"ID": str})
    sample = pd.read_csv(args.sample, dtype={"ID": str})
    required = ["ID", "share_added_vessel", "score_added_vessel"]
    assert test.columns.tolist() == ["ID"] + feature_cols
    assert sample.columns.tolist() == required
    assert len(test) == len(sample) == args.expected_rows
    assert (
        test.ID.is_unique
        and sample.ID.is_unique
        and test.ID.notna().all()
        and sample.ID.notna().all()
    )
    assert set(test.ID) == set(sample.ID)
    values = test[feature_cols].to_numpy(np.float32).reshape(-1, 15, 25)
    assert not np.isinf(values).any(), "Only finite values or missing NaN cells are valid inputs."
    observed = np.isfinite(values).any((1, 2))
    component = np.full((len(test), len(spec["component_order"])), summary["target_mean"])
    component[:, 2] = summary["conservative_target_mean"]
    evidence = np.full(len(test), summary["addition_rate"])
    if observed.any():
        component[observed], evidence[observed], _ = model.predict_components(values[observed])
    primary = np.clip(component @ np.asarray(spec["weights"]), 0, 1)
    alternative = np.clip(component @ np.asarray(spec["conservative_weights"]), 0, 1)
    primary[~observed] = summary["target_mean"]
    alternative[~observed] = summary["conservative_target_mean"]
    evidence = np.clip(evidence, 0, 1)
    records = {}
    for path, prediction in [(args.output, primary), (args.alternative, alternative)]:
        frame = (
            pd.DataFrame(
                {"ID": test.ID, "share_added_vessel": prediction, "score_added_vessel": evidence}
            )
            .set_index("ID")
            .loc[sample.ID]
            .reset_index()
        )
        assert frame.columns.tolist() == required and frame.ID.equals(sample.ID)
        arr = frame[required[1:]].to_numpy()
        assert np.isfinite(arr).all() and ((arr >= 0) & (arr <= 1)).all()
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, float_format="%.17g")
        saved = pd.read_csv(path, dtype={"ID": str})
        assert saved.ID.equals(sample.ID) and len(saved) == args.expected_rows
        assert np.isfinite(saved[required[1:]].to_numpy()).all()
        records[path.name] = {"rows": len(saved), "sha256": sha256(path)}
        print("Saved and validated", path, flush=True)
    models = {
        str(p.relative_to(BASE)): sha256(p)
        for directory in [model.tree_dir, model.neural_dir]
        for p in directory.iterdir()
        if p.suffix in [".joblib", ".cbm", ".pt", ".npy", ".npz"]
    }
    report = {
        "test_sha256": sha256(args.test),
        "sample_sha256": sha256(args.sample),
        "selection_sha256": sha256(args.selection),
        "models": models,
        "outputs": records,
        "wall_seconds": time.time() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "device": args.device,
        "test_used_for": "inference only; no fitting, selection, calibration, or training examples",
        "empty_window_fallback": "Training augmentation mean / addition rate; fixed before reading test.",
    }
    (args.output.parent / (args.output.stem + "_inference_report.json")).write_text(
        json.dumps(report, indent=2)
    )
    print(
        json.dumps(
            {
                "inference_seconds": report["wall_seconds"],
                "peak_memory_mb": report["peak_rss_mb"],
                "output_rows": len(test),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
