"""Reproduce the new CSV exactly without inspecting or tuning its predictions."""

import json, subprocess, sys, tempfile
from pathlib import Path
from simulation import BASE
from v2_predict import digest


def main():
    original = BASE.parent / "submission_v2.csv"
    report = json.loads(original.with_name("submission_v2_inference_report.json").read_text())
    assert digest(original) == report["output_sha256"]
    with tempfile.TemporaryDirectory(prefix="vessel-v2-reproduction-") as temp:
        output = Path(temp) / "submission_v2.csv"
        subprocess.run(
            [sys.executable, str(BASE / "v2_predict.py"), "--output", str(output)], check=True
        )
        assert original.read_bytes() == output.read_bytes()
    result = {
        "bit_for_bit_reproduced": True,
        "sha256": digest(original),
        "rows": report["rows"],
        "entrypoints": ["submission_inference_v2.py", "v2_predict.py"],
        "test_used_for": "saved-weight inference and reproduction only",
    }
    (BASE / "reports/v2_reproduction.json").write_text(json.dumps(result, indent=2))
    (BASE / "reports/v2_submission_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
