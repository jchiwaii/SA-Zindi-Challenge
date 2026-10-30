"""Record trained artifact sizes, hardware and code hashes for V2 review."""

import json, platform
import psutil
from simulation import BASE
from package_solution import digest


def main():
    runs = ["v2_development", "v2_neural_development", "v2_final", "v2_neural_final"]
    models, summaries = {}, {}
    for run in runs:
        directory = BASE / "runs" / run
        summaries[run] = json.loads((directory / "training_summary.json").read_text())
        for p in directory.iterdir():
            if p.suffix in [".joblib", ".pt", ".npz"]:
                models[str(p.relative_to(BASE))] = {"bytes": p.stat().st_size, "sha256": digest(p)}
    deployed_bytes = sum(
        v["bytes"]
        for k, v in models.items()
        if k.startswith(("runs/v2_final/", "runs/v2_neural_final/"))
    )
    record = {
        "models": models,
        "deployment_model_bytes": deployed_bytes,
        "training_runs": summaries,
        "hardware": {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "logical_cpu_count": psutil.cpu_count(),
            "ram_bytes": psutil.virtual_memory().total,
            "training_neural_device": "Apple MPS",
            "inference_device": "CPU",
        },
        "code_sha256": {p.name: digest(p) for p in sorted(BASE.glob("*.py"))},
    }
    (BASE / "reports/v2_model_inventory.json").write_text(json.dumps(record, indent=2))
    print("V2 deployment weights MB:", deployed_bytes / 1e6)


if __name__ == "__main__":
    main()
