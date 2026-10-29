"""Check saved inference on released audit data, without loading competition inputs."""

import argparse, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from audit_data import COLS
from simulation import BASE
from select_and_audit import Ensemble


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tree", default="final")
    p.add_argument("--neural", default="neural_final")
    args = p.parse_args()
    torch.set_num_threads(5)
    x = np.load(BASE / "reports/audit_examples.npz")["db"][:32]
    model = Ensemble(args.tree, args.neural, "cpu")
    pred, ev, _ = model.predict_components(x)
    assert np.isfinite(pred).all() and np.isfinite(ev).all()
    assert ((pred >= 0) & (pred <= 1)).all() and ((ev >= 0) & (ev <= 1)).all()
    reverse, reverse_ev, _ = model.predict_components(x[::-1].copy())
    np.testing.assert_allclose(pred, reverse[::-1], atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(ev, reverse_ev[::-1], atol=1e-12)
    independent = np.stack([model.predict_components(x[i : i + 1])[0][0] for i in [0, 5, 15, 31]])
    np.testing.assert_allclose(pred[[0, 5, 15, 31]], independent, atol=2e-6, rtol=2e-5)
    # Exercise the actual CSV entrypoint with missingness and reordered IDs.
    # These are released audit examples and artificial edge cases, never Test.csv.
    summary = json.loads((BASE / "runs" / args.tree / "training_summary.json").read_text())
    csv_contract_checked = False
    if "conservative_target_mean" in summary:
        with tempfile.TemporaryDirectory(prefix="vessel-inference-check-") as temporary:
            folder = Path(temporary)
            values = np.concatenate([x[:4], np.full((2, 15, 25), np.nan, np.float32)])
            values[-1, 0, 0] = 100.0
            ids = [f"LOCAL_CHECK_{i}" for i in range(len(values))]
            frame = pd.DataFrame(
                values.reshape(6, -1), columns=[f"m{m:02d}_{c}" for m in range(1, 16) for c in COLS]
            )
            frame.insert(0, "ID", ids)
            frame.to_csv(folder / "input.csv", index=False)
            pd.DataFrame(
                {"ID": ids[::-1], "share_added_vessel": 0.0, "score_added_vessel": 0.0}
            ).to_csv(folder / "sample.csv", index=False)
            subprocess.run(
                [
                    sys.executable,
                    str(BASE / "submission_inference.py"),
                    "--test",
                    str(folder / "input.csv"),
                    "--sample",
                    str(folder / "sample.csv"),
                    "--output",
                    str(folder / "primary.csv"),
                    "--alternative",
                    str(folder / "alternative.csv"),
                    "--tree",
                    args.tree,
                    "--neural",
                    args.neural,
                    "--expected-rows",
                    "6",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            for filename, expected in [
                ("primary.csv", summary["target_mean"]),
                ("alternative.csv", summary["conservative_target_mean"]),
            ]:
                saved = pd.read_csv(folder / filename)
                assert saved.ID.tolist() == ids[::-1]
                output = saved.iloc[:, 1:].to_numpy()
                assert np.isfinite(output).all() and ((output >= 0) & (output <= 1)).all()
                empty = saved.set_index("ID").loc[ids[4]]
                np.testing.assert_allclose(empty.share_added_vessel, expected, atol=1e-15)
                np.testing.assert_allclose(
                    empty.score_added_vessel, summary["addition_rate"], atol=1e-15
                )
            csv_contract_checked = True
    record = {
        "finite_and_bounded": True,
        "row_order_invariance": True,
        "per_window_independence": True,
        "max_row_order_difference": float(np.max(abs(pred - reverse[::-1]))),
        "max_single_window_difference": float(np.max(abs(pred[[0, 5, 15, 31]] - independent))),
        "test_acoustic_values_read": False,
        "csv_order_empty_window_and_single_cell_checked": csv_contract_checked,
    }
    (BASE / "reports/inference_tests.json").write_text(json.dumps(record, indent=2))
    print(record)


if __name__ == "__main__":
    main()
