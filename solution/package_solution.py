"""Package runnable code, trained weights and provenance without original challenge data."""

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
RUNS = ["development", "neural_development", "final", "neural_final"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def inventory():
    models = {}
    summaries = {}
    for run in RUNS:
        directory = BASE / "runs" / run
        summaries[run] = json.loads((directory / "training_summary.json").read_text())
        for p in directory.iterdir():
            if p.suffix in [".joblib", ".cbm", ".pt", ".npy", ".npz"]:
                models[str(p.relative_to(ROOT))] = {"bytes": p.stat().st_size, "sha256": digest(p)}
    code = {str(p.relative_to(ROOT)): digest(p) for p in sorted(BASE.glob("*.py"))}
    record = {
        "models": models,
        "model_bytes_total": sum(x["bytes"] for x in models.values()),
        "code_sha256": code,
        "training_runs": summaries,
    }
    (BASE / "reports/model_inventory.json").write_text(json.dumps(record, indent=2))
    original = json.loads((ROOT / "submission_inference_report.json").read_text())
    (BASE / "reports/submission_verification.json").write_text(json.dumps(original, indent=2))
    print("Inventoried model MB:", record["model_bytes_total"] / 1e6)


def package():
    files = [
        ROOT / ".gitignore",
        ROOT / "README.md",
        ROOT / "submission.csv",
        ROOT / "submission_conservative.csv",
        ROOT / "submission_inference_report.json",
    ]
    files += list(BASE.glob("*.py")) + list(BASE.glob("*.md")) + [BASE / "requirements.txt"]
    for p in (BASE / "reports").iterdir():
        if p.suffix in [".json", ".csv", ".png"] and p.name != "execution_status.json":
            files.append(p)
        elif p.name == "audit_predictions.csv.gz":
            files.append(p)
    for run in RUNS:
        files.extend(p for p in (BASE / "runs" / run).iterdir() if p.is_file())
    files.extend([BASE / "cache/background_windows.csv", BASE / "cache/reference_manifest.csv"])
    external = BASE / "external/vtuad_1.3.2"
    for name in [
        "CITATION.cff",
        "dataset_card.md",
        "license_evidence_v131.md",
        "registry_anonymized.csv",
        "selected_recordings.csv",
        "strict_interval_wav_SHA256SUMS.txt",
        "README_vtuad_strict_wav_vessel_parts.txt",
        "zenodo_record.json",
    ]:
        files.append(external / name)
    files.extend(p for p in (external / "minute_bands").iterdir() if p.suffix in [".npy", ".json"])
    files = sorted(set(files))
    for p in files:
        assert p.exists(), str(p)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    target = ROOT / "vessel_noise_submission_bundle.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT))
        z.writestr("COMMIT.txt", commit + "\n")
        z.writestr(
            "BUNDLE_MANIFEST.json",
            json.dumps(
                {
                    str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": digest(p)}
                    for p in files
                },
                indent=2,
            ),
        )
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        names = z.namelist()
        assert "Test.csv" not in names and "sample_submission.csv" not in names
        assert not any(
            n.startswith("South_African_Real_Acoustics/")
            or n.endswith((".wav", ".deflate", ".partial"))
            for n in names
        )
        assert "solution/cache/backgrounds.npy" not in names
    sha = digest(target)
    (ROOT / "vessel_noise_submission_bundle.sha256").write_text(f"{sha}  {target.name}\n")
    print(
        json.dumps(
            {
                "bundle": str(target),
                "files": len(files),
                "size_mb": target.stat().st_size / 1e6,
                "sha256": sha,
                "git_commit": commit,
            },
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory-only", action="store_true")
    args = p.parse_args()
    if args.inventory_only:
        inventory()
    else:
        package()


if __name__ == "__main__":
    main()
