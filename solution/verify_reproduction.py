"""Exactly reproduce saved submissions through the modular inference entrypoint."""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    names = ["submission.csv", "submission_conservative.csv"]
    original = json.loads((ROOT / "submission_inference_report.json").read_text())
    for name in names:
        assert digest(ROOT / name) == original["outputs"][name]["sha256"]
    with tempfile.TemporaryDirectory(prefix="vessel-reproduction-") as temporary:
        folder = Path(temporary)
        subprocess.run(
            [
                sys.executable,
                str(BASE / "predict.py"),
                "--output",
                str(folder / names[0]),
                "--alternative",
                str(folder / names[1]),
            ],
            check=True,
        )
        hashes = {}
        for name in names:
            assert (ROOT / name).read_bytes() == (folder / name).read_bytes(), name
            hashes[name] = digest(folder / name)
    record = {
        "bit_for_bit_reproduced": True,
        "original_entrypoint": "submission_inference.py",
        "reproduction_entrypoint": "predict.py",
        "device": "cpu",
        "test_used_for": "inference and output reproduction only",
        "sha256": hashes,
    }
    (BASE / "reports/reproduction_check.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
