"""Saved-weight V2 inference; no adaptation to competition inputs."""

import argparse, hashlib, json, resource, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from audit_data import COLS
from simulation import BASE
from v2_models import build_features
from neural import AcousticNet, neural_predict


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DirectModel:
    def __init__(self, run="v2_final"):
        self.directory = BASE / "runs" / run
        selected = self.directory / "combined_selection.json"
        if not selected.exists():
            selected = self.directory / "selection.json"
        self.spec = json.loads(selected.read_text())
        self.contract = json.loads((self.directory / "features.json").read_text())
        self.full = joblib.load(self.directory / "full.joblib")
        self.temporal = joblib.load(self.directory / "temporal.joblib")
        self.evidence = joblib.load(self.directory / "evidence.joblib")
        self.neural = None
        if "neural" in self.spec["components"]:
            self.neural_directory = BASE / "runs" / self.spec["neural_run"]
            checkpoint = torch.load(
                self.neural_directory / "model.pt", map_location="cpu", weights_only=True
            )
            self.neural = AcousticNet(checkpoint["n_features"])
            self.neural.load_state_dict(checkpoint["state_dict"])
            self.normalizer = dict(np.load(self.neural_directory / "normalizer.npz"))
        assert self.spec["test_used_for_fitting"] is False
        assert self.spec["evidence_gating"] is False

    def predict(self, values):
        valid = np.isfinite(values).any((1, 2))
        share = np.full(len(values), self.spec["target_mean"])
        evidence = np.full(len(values), self.spec["addition_rate"])
        if valid.any():
            f, names, columns = build_features(values[valid])
            assert names == self.contract["names"]
            assert columns.tolist() == self.contract["invariant_columns"]
            prediction = np.column_stack(
                [
                    np.clip(self.full.predict(f), 0, 1),
                    np.clip(self.temporal.predict(f[:, columns]), 0, 1),
                    np.full(len(f), self.spec["target_mean"]),
                ]
            )
            tree_evidence = self.evidence.predict_proba(f)[:, 1]
            if self.neural is not None:
                neural_share, neural_evidence, _, _ = neural_predict(
                    self.neural, values[valid], f, self.normalizer
                )
                prediction = np.column_stack([prediction, neural_share])
                if self.spec["evidence_source"] == "neural":
                    tree_evidence = neural_evidence
            share[valid] = prediction @ np.asarray(self.spec["weights"])
            evidence[valid] = tree_evidence
        return np.clip(share, 0, 1), np.clip(evidence, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, default=BASE.parent / "Test.csv")
    parser.add_argument("--sample", type=Path, default=BASE.parent / "sample_submission.csv")
    parser.add_argument("--output", type=Path, default=BASE.parent / "submission_v2.csv")
    parser.add_argument("--run", default="v2_final")
    parser.add_argument("--expected-rows", type=int, default=1500)
    args = parser.parse_args()
    start = time.time()
    torch.set_num_threads(4)
    model = DirectModel(args.run)
    test = pd.read_csv(args.test, dtype={"ID": str})
    sample = pd.read_csv(args.sample, dtype={"ID": str})
    columns = [f"m{m:02d}_{c}" for m in range(1, 16) for c in COLS]
    outputs = ["ID", "share_added_vessel", "score_added_vessel"]
    assert test.columns.tolist() == ["ID"] + columns and sample.columns.tolist() == outputs
    assert len(test) == len(sample) == args.expected_rows
    assert test.ID.is_unique and sample.ID.is_unique
    assert test.ID.notna().all() and sample.ID.notna().all()
    assert set(test.ID) == set(sample.ID)
    values = test[columns].to_numpy(np.float32).reshape(-1, 15, 25)
    assert not np.isinf(values).any()
    share, evidence = model.predict(values)
    result = pd.DataFrame(
        {"ID": test.ID, "share_added_vessel": share, "score_added_vessel": evidence}
    )
    result = result.set_index("ID").loc[sample.ID].reset_index()
    numeric = result[outputs[1:]].to_numpy()
    assert np.isfinite(numeric).all() and ((numeric >= 0) & (numeric <= 1)).all()
    assert result.ID.equals(sample.ID)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, float_format="%.17g")
    saved = pd.read_csv(args.output, dtype={"ID": str})
    assert saved.ID.equals(sample.ID)
    report = {
        "rows": len(saved),
        "output_sha256": digest(args.output),
        "test_sha256": digest(args.test),
        "sample_sha256": digest(args.sample),
        "models": {
            p.name: digest(p)
            for p in model.directory.iterdir()
            if p.suffix == ".joblib"
            or p.name in ["features.json", "selection.json", "combined_selection.json"]
        },
        "wall_seconds": time.time() - start,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "test_used_for": "saved-weight inference only",
    }
    if model.neural is not None:
        report["neural_models"] = {
            p.name: digest(p)
            for p in model.neural_directory.iterdir()
            if p.suffix in [".pt", ".npz"]
        }
    args.output.with_name(args.output.stem + "_inference_report.json").write_text(
        json.dumps(report, indent=2)
    )
    print("Saved", args.output, "rows", len(saved), "seconds", report["wall_seconds"], flush=True)


if __name__ == "__main__":
    main()
