"""Write a candid V2 comparison from local results, without competition inputs."""

import json, os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-vessel")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from simulation import BASE


def main():
    tree = json.loads((BASE / "reports/v2_audit_metrics.json").read_text())
    final = json.loads((BASE / "reports/v2_combined_audit_metrics.json").read_text())
    selection = json.loads((BASE / "runs/v2_development/combined_selection.json").read_text())
    rows = []
    for name, values in final.items():
        ci = values["v2_rmse_cluster_interval"]
        rows.append(
            f'| {name} | {values["old_rmse"]:.4f} | {tree[name]["v2_rmse"]:.4f} | '
            f'{values["v2_rmse"]:.4f} | {ci[0]:.4f}–{ci[2]:.4f} | {values["v2_auc"]:.4f} |'
        )
    text = f"""# Revision 2 transfer study — 30 October 2026

## The first attempt failed

The participant reported public scores **0.762574032** for the V1 primary and
**0.760998135** for its alternative: RMSE **0.237425967** and **0.239001864**.
Both received diagnostic AUC **0.463533333**. Those results contradict any
confidence that the original local simulation represented the competition.
The first submission files and weights are preserved.

These leaderboard figures were recorded as failure evidence. They were not used
to reconstruct targets or fit an output calibration. No V2 training, selection,
or augmentation design uses competition test inputs or their distributions.

## Findings and changes

The old feature importance strongly favored energy near 1–1.3 kHz, where the
public reference corpus and South African backgrounds have different spectral
shapes. This is a plausible transfer shortcut, not proof of the hidden test's
construction. Controlled changes to reference channels reproduce large errors
on released data. A separate issue was multiplying an unconditional fraction
regression by an addition classifier: detector errors then directly suppressed
the fraction. V2 predicts the fraction directly and keeps diagnostic evidence
separate.

The same licensed public recordings now receive four broader response families:
native, substantial tilt/curvature, transfer toward an independently drawn South
African spectral profile, and smooth channel responses. A fifth sinusoidal local
response is absent from training and used for validation and stress testing.
Recorded temporal variation is retained, with varied passage timing and shape.
Small background dither and independently selected export precision are applied
to both addition and no-addition examples. Their purpose is to prevent decimal
precision from becoming a presence cue.

The strength prior remains an assumption: 40% weak SNR from −30 to −7 dB,
40% uniform fractions 0.01–0.65, and 20% uniform fractions 0.25–0.85, followed by
25% no-addition controls. Targets use the common observation mask before export
rounding. This prior is not estimated from the competition data. Ambiguous weak
activity can still produce biased estimates, and transfer outside the simulated
range remains uncertain.

## Validation-selected model

Manually specified candidates are a full-feature LightGBM regressor, a restricted
temporal-feature LightGBM regressor, the previous physical-head neural architecture
retrained on the V2 generator, and the fixed training-mean baseline. The final
nonnegative blend was fitted only to validation predictions. Its components are
`{selection['components']}` and weights are `{selection['weights']}`.
Weighted validation RMSE is **{selection['weighted_validation_rmse']:.4f}**.
The evidence source is `{selection['evidence_source']}`; it does not multiply or
threshold the fraction prediction. No AutoML is used.

## Paired local comparison

Each row below evaluates 1,600 windows from the released audit station/periods and
public reference hulls excluded from development fitting. Within this comparison,
backgrounds, reference identities, temporal trajectories, masks and target
fractions are paired across channel families. No-addition observations are
identical. Intervals use the same two-way station-day/hull bootstrap as V1.

These historical partitions and results have already informed development. This
is a documented development stress study, **not an untouched confirmatory test**.
The comparisons concern development weights; final models are subsequently fitted
on all released partitions. The repeated cases across rows are not independent.

| Reference channel | V1 RMSE | V2 trees RMSE | V2 selected RMSE | Cluster 95% interval | V2 evidence AUC |
|---|---:|---:|---:|---|---:|
{chr(10).join(rows)}

Native-source regression can worsen while other channel cases improve. This
tradeoff and the full scenario table are retained; favorable cases alone are not
evidence of universal improvement. These numbers do not predict a Zindi score.
Actual V2 leaderboard performance remains unmeasured at delivery.

## Reproduction and limits

See `V2_README.md` for exact commands. Saved weights, feature contracts, training
manifests, seeds and dependency versions accompany the bundle. The single-file
inference entry point uses only public Python packages and performs no fitting.
The verification report records ID checks, bounds, per-window independence,
missing-window behavior and byte-for-byte reproduction. Training summaries record
local runtime and process memory; GPU allocations may not all appear in RSS.

All original data restrictions still apply. The external corpus/version is
unchanged; the disclosure draft describes the revised channel augmentation.
No natural vessel identity, ship-only energy, physical pressure, count, size or
calibrated vessel-presence probability has been established. A better local
synthetic study alone cannot establish those claims.
"""
    (BASE / "V2_REPORT.md").write_text(text)
    labels = list(final)
    position = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    ax.bar(position - 0.24, [final[k]["old_rmse"] for k in labels], width=0.24, label="V1")
    ax.bar(position, [tree[k]["v2_rmse"] for k in labels], width=0.24, label="V2 trees")
    ax.bar(position + 0.24, [final[k]["v2_rmse"] for k in labels], width=0.24, label="V2 selected")
    ax.set_xticks(position, [s.replace("_", " ") for s in labels])
    ax.set(ylabel="RMSE (lower is better)", title="Paired channel-transfer study on released data")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(BASE / "reports/v2_comparison.png", dpi=160)
    plt.close(fig)
    print("Wrote V2 report and comparison figure")


if __name__ == "__main__":
    main()
