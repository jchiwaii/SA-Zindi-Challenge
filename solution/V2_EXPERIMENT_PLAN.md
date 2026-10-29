# Revision 2: recover from failed transfer

The participant reported public scores 0.762574032 (primary) and 0.760998135
(alternative), with diagnostic AUC 0.463533333 for both. These results invalidate
the usefulness of the first local simulator as a predictor of competition
performance. They are recorded as failure evidence, never used to reconstruct
labels or calibrate per-example answers. Original submissions and weights remain
unchanged.

Before new fits, the following experiment is fixed:

1. Check old models on channel changes using only released training partitions
   and public audio. In particular, compare native Canadian spectra, substantial
   smooth response changes, and references equalized toward independently drawn
   South African spectra. All retain recorded reference temporal variation.
2. Train direct energy-fraction regressors across those response families, broad
   passage durations and strength priors. Break export-precision shortcuts by
   adding small independent measurement dither to backgrounds and applying the
   same export-rounding distribution to added and no-addition cases.
3. Compare a full-feature model with a restricted temporal-feature model. Fit a
   diagnostic classifier separately; do not multiply an unconditional fraction
   regressor by a second presence probability.
4. Select regularized regression blends on station/time and hull held-outs,
   reporting each generator family and strength regime separately. Entirely new
   source-channel constructions are stress cases, not assumed physical truth.
5. Report the old and new models on the same local cases. A local improvement is
   not a promised leaderboard improvement. Final refits and new inference use
   new filenames. No test-input inspection, fitting or calibration is allowed.

The original audit has already informed development; revision-2 results are a
new development study, not a claim that the historical audit remains untouched.
There are still no verified natural-vessel labels or validated physical-pressure
claims. No additional external corpus is introduced.
