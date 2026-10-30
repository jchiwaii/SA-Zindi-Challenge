"""Bundle V2 and its comparison provenance, excluding original challenge inputs."""

import hashlib, json, subprocess, zipfile
from pathlib import Path
from package_solution import digest

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
RUNS = [
    "development",
    "neural_development",
    "final",
    "neural_final",
    "v2_development",
    "v2_neural_development",
    "v2_final",
    "v2_neural_final",
]


def main():
    files = [
        ROOT / ".gitignore",
        ROOT / "README.md",
        ROOT / "submission_v2.csv",
        ROOT / "submission_v2_inference_report.json",
    ]
    files += list(BASE.glob("*.py")) + list(BASE.glob("*.md")) + [BASE / "requirements.txt"]
    files += [
        p
        for p in (BASE / "reports").iterdir()
        if p.suffix in [".json", ".csv", ".png"] and p.name != "execution_status.json"
    ]
    files += [p for p in (BASE / "reports").glob("*audit_predictions.csv.gz")]
    for run in RUNS:
        files += [p for p in (BASE / "runs" / run).iterdir() if p.is_file()]
    files += [BASE / "cache/background_windows.csv", BASE / "cache/reference_manifest.csv"]
    external = BASE / "external/vtuad_1.3.2"
    files += [
        external / name
        for name in [
            "CITATION.cff",
            "dataset_card.md",
            "license_evidence_v131.md",
            "registry_anonymized.csv",
            "selected_recordings.csv",
            "strict_interval_wav_SHA256SUMS.txt",
            "README_vtuad_strict_wav_vessel_parts.txt",
            "zenodo_record.json",
        ]
    ]
    files += [p for p in (external / "minute_bands").iterdir() if p.suffix in [".npy", ".json"]]
    files = sorted(set(files))
    assert all(p.exists() for p in files)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in files
    }
    destination = ROOT / "vessel_noise_v2_bundle.zip"
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for p in files:
            archive.write(p, p.relative_to(ROOT))
        archive.writestr("COMMIT.txt", commit + "\n")
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            assert name not in [
                "Test.csv",
                "sample_submission.csv",
                "solution/cache/backgrounds.npy",
            ]
            assert not name.startswith("South_African_Real_Acoustics/")
            assert not name.endswith((".wav", ".deflate", ".partial"))
    checksum = digest(destination)
    destination.with_suffix(".sha256").write_text(checksum + "  " + destination.name + "\n")
    print(
        json.dumps(
            {
                "bundle": str(destination),
                "mb": destination.stat().st_size / 1e6,
                "sha256": checksum,
                "git_commit": commit,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
