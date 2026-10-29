"""Produce training-only spectral QA figures and source coverage tables."""

import os, json, warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-vessel")
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simulation import BASE


def main():
    out = BASE / "reports"
    out.mkdir(exist_ok=True)
    meta = pd.read_csv(BASE / "cache/background_windows.csv")
    bg = np.load(BASE / "cache/backgrounds.npy", mmap_mode="r")
    rng = np.random.default_rng(31415)
    frequency = 10 ** (1.5 + 0.1 * np.arange(25))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for site, g in meta[meta.split == "train"].groupby("site"):
        idx = rng.choice(g.index, min(len(g), 2000), replace=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(bg[idx], axis=(0, 1))
        axes[0].plot(frequency, median, label=f"Site {site}")
    axes[0].set(
        xscale="log",
        xlabel="Band centre (Hz)",
        ylabel="Export band level (dB; calibration unverified)",
        title="Released South African training spectra",
    )
    axes[0].legend(ncols=2, fontsize=8)
    manifest = BASE / "cache/reference_manifest.csv"
    if manifest.exists():
        refs = pd.read_csv(manifest)
    else:
        refs = pd.DataFrame(
            [
                json.loads(p.read_text())
                for p in (BASE / "external/vtuad_1.3.2/minute_bands").glob("*.json")
            ]
        )
    for label, g in refs[refs.split == "train"].groupby("label"):
        spectra = np.array([np.mean(np.load(BASE / p), axis=0) for p in g.feature_path])
        axes[1].plot(frequency, np.median(spectra, axis=0), label=label)
    axes[1].set(
        xscale="log",
        xlabel="Band centre (Hz)",
        ylabel="Relative band energy (dB)",
        title="Public vessel-reference spectra (training hulls only)",
    )
    axes[1].legend(fontsize=8, ncols=2)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(out / "training_spectra.png", dpi=160)
    plt.close(fig)
    refs.groupby(["split", "label"]).agg(
        recordings=("path", "size"),
        hulls=("hull_id", "nunique"),
        minutes=("complete_minutes", "sum"),
    ).to_csv(out / "external_coverage.csv")
    meta.groupby(["split", "site"]).agg(
        windows=("window_id", "size"), mean_observed_cells=("observed_cells", "mean")
    ).to_csv(out / "background_coverage.csv")
    print("Saved", out / "training_spectra.png")


if __name__ == "__main__":
    main()
