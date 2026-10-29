# Underwater vessel-noise submission

This is a local, open-source modelling pipeline for the Zindi underwater vessel-noise challenge. It estimates **controlled added-reference energy**, not all natural vessel noise. A leaderboard rank is not guaranteed or verified.

## Deliverables

- `../submission.csv`: primary validation-selected ensemble, 1,500 test IDs.
- `../submission_conservative.csv`: alternative using a lower-addition training prior.
- `VALIDATION_REPORT.md`: independent audit, model comparison, source coverage, assumptions and limitations.
- `EXTERNAL_DATA_DISCLOSURE.md`: required external-data declaration, prepared for posting by the participant; it has not been posted automatically.
- `runs/final/` and `runs/neural_final/`: final trained models and training records.
- `reports/frozen_selection.json`: validation-selected blend frozen before any competition test acoustic values are read.
- `../submission_inference_report.json`: input/model/output hashes, inference time and memory.
- `../vessel_noise_submission_bundle.zip`: runnable code, trained weights, predictions and provenance; original challenge inputs are excluded.
- `reports/reproduction_check.json`: confirms both CSVs were reproduced byte for byte through the modular inference entry point.

The two CSVs express different assumptions about unknown addition strength. The primary is selected by the predeclared validation objective; the alternative is documented sensitivity analysis. No test-distribution fitting or leaderboard probing is used.

## Reproduce the CSVs using saved weights

Place `Test.csv` and `sample_submission.csv` in the parent directory of `solution/`. From that parent directory:

```bash
python3 -m pip install -r solution/requirements.txt
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=5 python3 solution/submission_inference.py
```

The single-file entry point requires no custom package installation. Inference runs on CPU without internet access. No training data or external WAV files are needed for this command. To choose explicit input and output locations:

```bash
python3 solution/submission_inference.py --test /path/Test.csv --sample /path/sample_submission.csv --output /path/submission.csv --alternative /path/submission_conservative.csv
```

The script validates IDs, column order, row count, finite values and [0,1] bounds. Missing measurements stay missing. Completely unobserved windows, if any, receive a fixed training-mean fallback. It does not fit parameters or alter saved models.

## Rebuild training data

Keep the original challenge folder `South_African_Real_Acoustics/` in the parent directory. Only use these restricted measurements for this challenge.

```bash
python3 solution/audit_data.py
python3 solution/prepare_backgrounds.py
python3 solution/download_external.py --metadata-only
OPENBLAS_NUM_THREADS=1 python3 solution/prepare_external.py
python3 solution/test_physics.py
```

External data are public VTUAD v1.3.2, CC BY 4.0. The downloader deterministically selects 201 recordings from 172 hulls and verifies the provider's original WAV SHA-256 before converting each recording. About 1.23 GB of compressed audio is fetched the first time; later runs reuse the local cache. A complete-minute feature cache and provenance can be retained to avoid another download. See the disclosure file for exact access instructions and attribution.

## Reproduce development and selection

These are explicitly specified models, not AutoML. The development models exclude the declared station/time audit partitions and external audit hulls.

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=6 python3 solution/train.py --name development --n-train 100000 --n-val 2500 --iterations 2000
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=5 python3 solution/neural.py --name neural_development --templates development --n-train 80000 --n-val 2500 --epochs 60
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=5 python3 solution/select_and_audit.py
```

`neural.py` uses Apple MPS where available and otherwise CPU; `--device cpu` forces CPU. GPU training may have minor nondeterminism despite fixed seeds. Saved weights plus CPU inference reproduce the delivered predictions. The selected epoch and final training arguments are recorded in each run's `config.json` and `training_summary.json`.

`reports/frozen_selection.json` is saved **before** the independent audit runs. Do not retune using audit results while continuing to call that audit untouched. No competition test values are used in this process.

## Final training and verification

`run_final.py` reads the chosen iteration/epoch counts and refits on all permitted training partitions. It does not read the competition test. `submission_inference.py` and its modular equivalent `predict.py` are the only modelling-stage entry points that read test acoustic values.

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=6 python3 solution/run_final.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=5 python3 solution/test_inference.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=5 python3 solution/submission_inference.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=5 python3 solution/verify_reproduction.py
python3 solution/make_report.py
python3 solution/package_solution.py --inventory-only
```

`test_inference.py` uses the local audit arrays produced by `select_and_audit.py`; these raw arrays are omitted from the distribution. It checks row-order and per-window independence, missing-window handling, single-cell inputs and the CSV contract. `verify_reproduction.py` uses only saved models and competition inputs for inference, comparing both CSVs byte for byte without changing the delivered files. After committing code, `python3 solution/package_solution.py` builds and checks the ZIP with the Git revision and a SHA-256 manifest.

Inspect `VALIDATION_REPORT.md` for the exact data construction, baseline comparisons, stress scenarios, clustering of uncertainty, and limitations. The public-source disclosure must be posted through the competition discussion board as required by the rules. Uploading the submission and posting the disclosure are left to the participant; neither action has been performed automatically.
