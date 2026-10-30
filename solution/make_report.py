"""Write the evidence report from saved results; no competition test-data analysis."""

import json, os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-vessel")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simulation import BASE
from train import SCENARIOS


def read(path):
    return json.loads(path.read_text())


def main():
    out = BASE / "reports"
    audit = read(out / "audit_metrics.json")
    selection = read(out / "frozen_selection.json")
    tree = read(BASE / "runs/development/validation_metrics.json")
    neural = read(BASE / "runs/neural_development/validation_metrics.json")
    predictions = pd.read_csv(out / "audit_predictions.csv.gz")
    site_rows = []
    for (scenario, site), group in predictions.groupby(["scenario", "site"]):
        error = group.prediction - group.share
        site_rows.append(
            dict(
                scenario=scenario,
                site=site,
                n=len(group),
                rmse=float(np.sqrt(np.mean(error**2))),
                bias=float(error.mean()),
            )
        )
    pd.DataFrame(site_rows).to_csv(out / "audit_by_site.csv", index=False)
    standard_subset = predictions[predictions.scenario == "standard"].iloc[:1200]
    subset_rmse = np.sqrt(np.mean((standard_subset.prediction - standard_subset.share) ** 2))
    control_rows = [f"| Unmodified comparison subset | {subset_rmse:.4f} | 0.0000 |"]
    for name, values in audit["standard"]["sensitivity_controls"].items():
        control_rows.append(
            f'| {name.replace("_", " ")} | {values["rmse"]:.4f} | '
            f'{values["mean_absolute_prediction_change"]:.4f} |'
        )
    rows = []
    for s in SCENARIOS:
        ci = audit[s]["rmse_cluster_bootstrap_95_interval"]
        rows.append(
            f'| {s} | {tree["constant"][s]["rmse"]:.4f} | {tree["received_energy"][s]["rmse"]:.4f} | {tree["lightgbm"][s]["rmse"]:.4f} | {tree["catboost"][s]["rmse"]:.4f} | {neural[s]["rmse"]:.4f} | {audit[s]["primary"]["rmse"]:.4f} | {ci[0]:.4f}–{ci[2]:.4f} |'
        )
    weights = ", ".join(
        f"{name}: {weight:.4f}"
        for name, weight in zip(selection["component_order"], selection["weights"])
    )
    summaries = []
    for name in ["development", "neural_development", "final", "neural_final"]:
        p = BASE / "runs" / name / "training_summary.json"
        if p.exists():
            info = read(p)
            summaries.append(
                f'| {name} | {info["n_train"]:,} | {info["wall_seconds"]/60:.1f} | {info["peak_rss_mb"]:.0f} |'
            )
    text = f"""# Vessel-addition submission: validation and reproducibility report

**Historical V1 local evaluation.** Subsequent participant-reported public RMSE
was 0.237425967 (primary) and 0.239001864 (alternative), with evidence AUC
0.463533333. The original generator did not establish reliable transfer. See
`V2_REPORT.md` for the revised study; the original local numbers below are retained.

## Status and meaning of the results

This package estimates the fraction of measured energy due to an added vessel-passage reference. It does **not** estimate all naturally occurring vessel energy, vessel count, vessel size, calibrated physical pressure or a calibrated probability of natural vessel presence. No leaderboard score or winning rank has been verified.

All performance numbers below come from our own controlled additions to held-out released measurements. Their addition strengths, propagation responses and temporal envelopes are assumptions declared in the code; they are not the undisclosed competition generator. These scores cannot be converted into a promised leaderboard score. The test acoustic values were not read for training, tuning, calibration, augmentation design or model selection.

## Data and splits

- Original release: 1,715,779 unlabelled rows from ten South African sites. Original files are preserved.
- Derived backgrounds: 133,269 fixed, non-overlapping 15-minute recorder windows. A retained window has at least 125 observed minute-band cells. Invalid durations, sentinel-like levels and near-flat records are screened before window construction; the flatness filter is a heuristic and may exclude usable observations. Gaps remain missing. The acquisition-rate token is not used to override finite upper-band measurements because its interpretation is unresolved.
- Timestamp handling explicitly converts Parquet timestamp units to nanoseconds before obtaining minute indices. All clocks remain in their exported convention; no UTC or AIS alignment is claimed.
- Development: within each site and recorder, the first approximately 70% of days are used for training, the next 15% for validation and the last 15% for audit. Very short deployments use six-hour blocks. Durban (site 2) is an entire validation station; Saldanha Bay (site 13) is an entire audit station.
- Public references: 201 checksum-verified strict-interval recordings, 172 hulls and 661 complete minutes from VTUAD v1.3.2. Public hull hashes assign training, validation and audit partitions; no hull or exact WAV crosses a partition. Labels are AIS-derived, not verified vessel-only audio. The exact deterministic subset and hashes are supplied.
- Background sampling uses square-root station balancing. Final deployment models are refitted using all released background partitions and all selected reference partitions, **after** the development selection and audit. The audit metrics assess the development models, not an independently tested final refit.

Every simulated example has a saved background window identifier, reference provenance, seed, scenario and target. See `cache/background_windows.csv`, `cache/reference_manifest.csv`, and the compressed per-run example manifests.

## Acoustic processing and training examples

The public audio is integrated into the documented 25 base-10 third-octave bands using two-second Hann periodograms and linear averaging over complete minutes. Absolute digital gain is removed; no pressure calibration is assumed. An independent 1,000 Hz tone check verifies band placement.

Augmentations stretch/crop or reverse reference trajectories, apply smooth frequency response changes, and vary passage envelopes. The standard strength prior is 80% uniform SNR from −30 to 0 dB and 20% uniform SNR from −5 to +8 dB, with 25% no-addition controls. SNR is relative to measured background energy. This prior was chosen without test inputs and is uncertain. Observation masks are applied independently; the energy fraction is then recalculated using exactly the same observed cells in numerator and denominator. Naturally occurring vessels can remain in every background, including no-addition controls.

Six scenarios test the standard generator, weak additions, strong additions, larger channel changes, heavier missingness and flat temporal envelopes. They are detailed in `simulation.py`. A separate conservative regressor weights training examples by `exp(-4 * share)` to examine a lower-addition prior. It is an explicit sensitivity alternative, not a correction inferred from the test distribution.

## Models and frozen selection

The manually specified models are LightGBM, CatBoost and a compact convolutional neural network. The tree models use 555 spectral, temporal, coverage and energy-summary features, including distances to 48 broad background spectral clusters fitted only on training backgrounds. No source ID or nearest-recording fingerprint is a predictor. The neural model combines these features with a 15×25 acoustic representation and an energy-weighted cell-fraction head. A separate LightGBM classifier provides continuous evidence for an added reference; its output is not a natural-vessel probability.

No AutoML or automated architecture search is used. Standard early stopping and a constrained non-negative blend are fitted on validation data only. The frozen weights are **{weights}**. The blend considers both raw regressors and regressors multiplied by continuous addition evidence. This additional combination was motivated by improved validation error, without hard presence thresholds. The separate conservative file uses the conservative regressor multiplied by the same evidence score.

## Evaluation

Validation model columns and the independent audit ensemble columns use different held-out partitions. They must not be interpreted as a single common test comparison. Audit intervals use a two-way Poisson cluster bootstrap over station-days and reference hulls, acknowledging repeated measurements and reused passage recordings. There are {audit['n_examples']:,} audit examples, {audit['background_windows']:,} distinct backgrounds and {audit['source_hulls']} reference hulls. Source diversity remains limited, so the intervals do not cover every form of domain uncertainty.

| Scenario | Val constant | Val energy baseline | Val LightGBM | Val CatBoost | Val neural | Audit ensemble | Audit RMSE 95% interval |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

The predeclared weighted mean of audit scenario RMSEs is {audit['weighted_rmse']:.4f}. Do not interpret `1 − this value` as a Zindi score.

Detailed no-addition, weak, medium and strong fraction results, continuous-evidence AUC, signed biases and perturbation sensitivity are in `reports/audit_metrics.json`. Frequency reversal and time shuffling are sensitivity checks, not evidence of natural vessel identity. The model's response to a uniform gain shift exposes dependence on acquisition conditions. Rounding to 0.1 and 0.5 dB checks sensitivity to numerical precision.

The controls below use the same first 1,200 standard audit examples. Rounding measurably changes predictions, indicating limited precision robustness. These controls were evaluated after selection and did not alter the frozen recipe. Station-level sample sizes, errors and signed biases are available in `reports/audit_by_site.csv`.

| Fixed perturbation | RMSE | Mean absolute prediction change |
|---|---:|---:|
{chr(10).join(control_rows)}

## Reproducibility and compute

Random seeds, model parameters, exact selected sources, dependency versions, trained weights and inference checksums are included. Training used local Apple Silicon hardware with 11 CPU cores, 18 GB RAM, macOS and the Apple MPS GPU for the neural network. Final inference runs on CPU. Seeded GPU training may exhibit floating-point nondeterminism; the saved CPU inference artifacts are the reproducibility authority.

| Run | Training examples | Wall minutes | Process peak RSS MiB |
|---|---:|---:|---:|
{chr(10).join(summaries)}

Per-process RSS does not necessarily include every GPU allocation. Model sizes, final inference runtime, memory and output hashes are recorded by the packaging and inference scripts. Numerical tests cover linear energy arithmetic, mask consistency, correct frequency band placement, normalization invariance, split isolation and seeded augmentation repeatability.

## Limitations and research requirements

Separating an added reference from an arbitrary unlabelled background is not always identifiable. The strongest remaining uncertainties are the undisclosed addition distribution, transfer of Canadian spectral patterns to South African recorders, missing observations, natural interfering sounds and the relatively small number of independent reference hulls. A reference may contain ambient energy; the target is not isolated ship-only energy.

The source audit plots and held-out examples are diagnostic acoustic evidence. They are **not** a validated pressure map or a completed research-prize submission. A credible natural-vessel monitoring claim requires independent acoustic annotations or calibrated deployments, relevant interference controls and correctly aligned external evidence. Any future map should be labelled as relative acoustic activity until independently calibrated, and should show coverage, observation dates and station-coordinate uncertainty.

## External-data declaration

The draft `EXTERNAL_DATA_DISCLOSURE.md` supplies source URLs, the exact version, access instructions, licence attribution and preprocessing. It has **not** been posted on your behalf. Publish the required declaration in the challenge discussion board before relying on external-data eligibility. The public discussion route showed the information page while signed out, so participant discussions were not reviewed. No challenge data or private code was sent to other participants, and no competition submission was uploaded.

Sources: [competition](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge), [VTUAD v1.3.2](https://doi.org/10.5281/zenodo.23037895), [Ocean Networks Canada](https://www.oceannetworks.ca/data/).
"""
    (BASE / "VALIDATION_REPORT.md").write_text(text)
    # Clear held-out performance plots, with no test-distribution inspection.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    std = predictions[predictions.scenario == "standard"]
    axes[0].hexbin(std.share, std.prediction, gridsize=35, mincnt=1, cmap="viridis")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(
        xlabel="Known added energy fraction",
        ylabel="Estimated fraction",
        title="Untouched standard audit",
    )
    means = (
        predictions.groupby("scenario")
        .apply(lambda x: np.sqrt(np.mean((x.prediction - x.share) ** 2)), include_groups=False)
        .reindex(SCENARIOS)
    )
    axes[1].barh(SCENARIOS, means, color="#24788f")
    axes[1].set(xlabel="RMSE (lower is better)", title="Independent stress scenarios")
    axes[1].invert_yaxis()
    grouped = (
        std.assign(bin=pd.cut(std.share, [-0.001, 0, 0.02, 0.05, 0.1, 0.2, 0.4, 1]))
        .groupby("bin", observed=True)
        .agg(target=("share", "mean"), predicted=("prediction", "mean"), n=("share", "size"))
    )
    axes[2].plot(grouped.target, grouped.predicted, "o-", label="Mean estimate")
    axes[2].plot([0, 1], [0, 1], "k--", lw=1)
    axes[2].set(
        xlabel="Mean true share in bin",
        ylabel="Mean prediction",
        title="Audit calibration by energy fraction",
    )
    fig.savefig(out / "audit_performance.png", dpi=160)
    plt.close(fig)
    if not (out / "audit_examples.npz").exists():
        print(
            "Report complete. Raw audit arrays are omitted from the distribution; existing example plots are retained."
        )
        return
    examples = np.load(out / "audit_examples.npz")
    x = examples["db"]
    y = examples["target"]
    p = examples["prediction"]
    indexes = [
        int(np.argmin(y)),
        int(np.argmin(abs(y - 0.04))),
        int(np.argmin(abs(y - 0.2))),
        int(np.argmax(abs(y - p))),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for ax, i in zip(axes.flat, indexes):
        im = ax.imshow(x[i].T, aspect="auto", origin="lower", cmap="magma")
        ax.set(
            title=f"Audit example {i}: true {y[i]:.3f}, estimate {p[i]:.3f}",
            xlabel="Relative minute",
            ylabel="Frequency band (low → high)",
        )
        fig.colorbar(im, ax=ax, label="Export dB (uncalibrated)")
    fig.savefig(out / "audit_examples.png", dpi=150)
    plt.close(fig)
    print("Wrote evidence report and audit figures.")


if __name__ == "__main__":
    main()
