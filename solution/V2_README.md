# Revised submission

The first submission failed on the public leaderboard. Read `V2_REPORT.md` for
the reported results, controlled failure tests, improvements and remaining
uncertainty. The revised output is `../submission_v2.csv`; previous submissions
remain unchanged. V2 has not been uploaded automatically.

## Saved-weight inference

Keep `Test.csv` and `sample_submission.csv` in the parent directory of `solution/`.
Unpack the V2 bundle so its model directories remain under `solution/runs/`.

```bash
python3 -m pip install -r solution/requirements.txt
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 python3 solution/submission_inference_v2.py
```

The single-file entry point needs no custom package installation. It validates all
1,500 IDs, the exact schema and finite [0,1] outputs. Empty windows receive a
training-only mean. Test inputs are used only for inference. The modular
equivalent is `solution/v2_predict.py`. Both accept `--test`, `--sample`, and
`--output` paths. The source-data declaration is still required by the rules;
no public post has been made on your behalf.

## Reproduce training and development comparisons

Use the original README's data-preparation commands first. The public corpus and
source selection are unchanged. No additional external download is necessary if
the included minute-band cache is available.

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 python3 solution/v2_diagnose.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=6 python3 solution/v2_train.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 python3 solution/v2_neural.py
python3 solution/v2_select.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 python3 solution/v2_audit.py --label v2_combined
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=6 python3 solution/v2_train.py --final --n-train 160000
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 python3 solution/v2_neural.py --final
python3 solution/v2_finalize.py
python3 solution/v2_make_standalone.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 python3 solution/v2_test.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 python3 solution/submission_inference_v2.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 python3 solution/v2_verify.py
python3 solution/v2_report.py
python3 solution/v2_inventory.py
```

Neural training defaults to Apple's MPS GPU; use `--device cpu` on other systems.
The architecture, seeds and schedules are fixed. GPU training can have numerical
nondeterminism; the delivered saved weights and CPU inference define the exact
submission. `v2_finalize.py` transfers validation-selected blend settings to the
final model paths without inspecting test data. The original tree-only comparison
is retained separately; reproduced combined comparisons are development evidence.

After committing code, `python3 solution/v2_package.py` builds the local bundle
and its checksum. The bundle omits original challenge inputs, raw WAVs and raw
South African acoustic arrays. It contains trained weights, scripts, manifests,
public-source minute features, licence/source notices and evaluation records.
