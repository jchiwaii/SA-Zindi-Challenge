"""Physics, paired interventions, and production CSV edge-case tests."""

import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from acoustic_model import energy
from audit_data import COLS
from simulation import BASE
from v2_simulation import DiverseSimulator
from v2_predict import DirectModel


def main():
    torch.set_num_threads(4)
    sim = DiverseSimulator(["audit_time", "audit_site"], ["audit"])
    x, y, meta, cells = sim.generate(
        96, 65817, family="native", paired_channels=True, return_cells=True
    )
    changed, same_y, same_meta = sim.generate(
        96, 65817, family="local_channel", paired_channels=True
    )
    np.testing.assert_allclose(y, same_y, atol=2e-7)
    assert meta.window_id.equals(same_meta.window_id)
    assert meta.reference_member.equals(same_meta.reference_member)
    assert np.array_equal(np.isnan(x), np.isnan(changed))
    np.testing.assert_array_equal(x[~meta.is_added], changed[~same_meta.is_added])
    e, _ = energy(x)
    np.testing.assert_allclose((e * cells).sum((1, 2)) / e.sum((1, 2)), y, atol=0.006, rtol=0)
    model = DirectModel()
    share, evidence = model.predict(x)
    reverse, rev_evidence = model.predict(x[::-1].copy())
    np.testing.assert_allclose(share, reverse[::-1], atol=2e-6, rtol=1e-5)
    np.testing.assert_allclose(evidence, rev_evidence[::-1], atol=2e-6, rtol=1e-5)
    isolated = np.array([model.predict(x[i : i + 1])[0][0] for i in [0, 11, 43]])
    np.testing.assert_allclose(share[[0, 11, 43]], isolated, atol=2e-6, rtol=1e-5)
    with tempfile.TemporaryDirectory(prefix="vessel-v2-check-") as temp:
        folder = Path(temp)
        raw = np.concatenate([x[:4], np.full((2, 15, 25), np.nan, dtype=np.float32)])
        raw[-1, 0, 0] = 90
        ids = [f"LOCAL_ONLY_{i}" for i in range(6)]
        columns = [f"m{m:02d}_{c}" for m in range(1, 16) for c in COLS]
        inputs = pd.DataFrame(raw.reshape(6, -1), columns=columns)
        inputs.insert(0, "ID", ids)
        inputs.to_csv(folder / "input.csv", index=False)
        pd.DataFrame(
            {"ID": ids[::-1], "share_added_vessel": 0.0, "score_added_vessel": 0.0}
        ).to_csv(folder / "sample.csv", index=False)
        subprocess.run(
            [
                sys.executable,
                str(BASE / "submission_inference_v2.py"),
                "--test",
                str(folder / "input.csv"),
                "--sample",
                str(folder / "sample.csv"),
                "--output",
                str(folder / "output.csv"),
                "--expected-rows",
                "6",
            ],
            check=True,
        )
        output = pd.read_csv(folder / "output.csv")
        assert output.ID.tolist() == ids[::-1]
        values = output.iloc[:, 1:].to_numpy()
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        np.testing.assert_allclose(
            output.set_index("ID").loc[ids[4], "share_added_vessel"],
            model.spec["target_mean"],
            atol=1e-15,
        )
    record = {
        "physics_and_mask": True,
        "paired_channels": True,
        "row_order_and_per_window": True,
        "csv_ids_missing_and_single_cell": True,
        "competition_inputs_read": False,
    }
    (BASE / "reports/v2_tests.json").write_text(json.dumps(record, indent=2))
    print(record)


if __name__ == "__main__":
    main()
