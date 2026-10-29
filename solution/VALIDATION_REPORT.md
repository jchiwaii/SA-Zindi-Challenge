# Vessel-addition submission: validation and reproducibility report

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

No AutoML or automated architecture search is used. Standard early stopping and a constrained non-negative blend are fitted on validation data only. The frozen weights are **lightgbm: 0.0000, catboost: 0.0000, conservative: 0.0000, neural: 0.0000, lightgbm_evidence_weighted: 0.3161, catboost_evidence_weighted: 0.0365, conservative_evidence_weighted: 0.0000, neural_evidence_weighted: 0.6473**. The blend considers both raw regressors and regressors multiplied by continuous addition evidence. This additional combination was motivated by improved validation error, without hard presence thresholds. The separate conservative file uses the conservative regressor multiplied by the same evidence score.

## Evaluation

Validation model columns and the independent audit ensemble columns use different held-out partitions. They must not be interpreted as a single common test comparison. Audit intervals use a two-way Poisson cluster bootstrap over station-days and reference hulls, acknowledging repeated measurements and reused passage recordings. There are 15,000 audit examples, 10,792 distinct backgrounds and 21 reference hulls. Source diversity remains limited, so the intervals do not cover every form of domain uncertainty.

| Scenario | Val constant | Val energy baseline | Val LightGBM | Val CatBoost | Val neural | Audit ensemble | Audit RMSE 95% interval |
|---|---:|---:|---:|---:|---:|---:|---|
| standard | 0.2239 | 0.1664 | 0.0817 | 0.0839 | 0.0765 | 0.0946 | 0.0816–0.1085 |
| weak | 0.1344 | 0.0969 | 0.0579 | 0.0606 | 0.0515 | 0.0820 | 0.0657–0.0963 |
| strong | 0.4135 | 0.3001 | 0.1215 | 0.1240 | 0.1096 | 0.1383 | 0.1134–0.1693 |
| channel_shift | 0.2305 | 0.1813 | 0.0885 | 0.0911 | 0.0811 | 0.1065 | 0.0918–0.1226 |
| missing | 0.2342 | 0.1769 | 0.0943 | 0.0967 | 0.0868 | 0.0994 | 0.0858–0.1168 |
| flat | 0.2246 | 0.1603 | 0.0853 | 0.0864 | 0.0817 | 0.1094 | 0.0953–0.1231 |

The predeclared weighted mean of audit scenario RMSEs is 0.0982. Do not interpret `1 − this value` as a Zindi score.

Detailed no-addition, weak, medium and strong fraction results, continuous-evidence AUC, signed biases and perturbation sensitivity are in `reports/audit_metrics.json`. Frequency reversal and time shuffling are sensitivity checks, not evidence of natural vessel identity. The model's response to a uniform gain shift exposes dependence on acquisition conditions. Rounding to 0.1 and 0.5 dB checks sensitivity to numerical precision.

## Reproducibility and compute

Random seeds, model parameters, exact selected sources, dependency versions, trained weights and inference checksums are included. Training used local Apple Silicon hardware with 11 CPU cores, 18 GB RAM, macOS and the Apple MPS GPU for the neural network. Final inference runs on CPU. Seeded GPU training may exhibit floating-point nondeterminism; the saved CPU inference artifacts are the reproducibility authority.

| Run | Training examples | Wall minutes | Process peak RSS MiB |
|---|---:|---:|---:|
| development | 100,000 | 8.3 | 1073 |
| neural_development | 80,000 | 13.7 | 1150 |

Per-process RSS does not necessarily include every GPU allocation. Model sizes, final inference runtime, memory and output hashes are recorded by the packaging and inference scripts. Numerical tests cover linear energy arithmetic, mask consistency, correct frequency band placement, normalization invariance, split isolation and seeded augmentation repeatability.

## Limitations and research requirements

Separating an added reference from an arbitrary unlabelled background is not always identifiable. The strongest remaining uncertainties are the undisclosed addition distribution, transfer of Canadian spectral patterns to South African recorders, missing observations, natural interfering sounds and the relatively small number of independent reference hulls. A reference may contain ambient energy; the target is not isolated ship-only energy.

The source audit plots and held-out examples are diagnostic acoustic evidence. They are **not** a validated pressure map or a completed research-prize submission. A credible natural-vessel monitoring claim requires independent acoustic annotations or calibrated deployments, relevant interference controls and correctly aligned external evidence. Any future map should be labelled as relative acoustic activity until independently calibrated, and should show coverage, observation dates and station-coordinate uncertainty.

## External-data declaration

The draft `EXTERNAL_DATA_DISCLOSURE.md` supplies source URLs, the exact version, access instructions, licence attribution and preprocessing. It has **not** been posted on your behalf. Publish the required declaration in the challenge discussion board before relying on external-data eligibility. The public discussion route showed the information page while signed out, so participant discussions were not reviewed. No challenge data or private code was sent to other participants, and no competition submission was uploaded.

Sources: [competition](https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge), [VTUAD v1.3.2](https://doi.org/10.5281/zenodo.23037895), [Ocean Networks Canada](https://www.oceannetworks.ca/data/).
