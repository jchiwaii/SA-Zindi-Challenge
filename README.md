# Underwater vessel-noise challenge

A reproducible local pipeline for estimating the controlled vessel-reference energy fraction in the South African underwater-noise challenge.

The original submissions failed on the public leaderboard (reported RMSE 0.2374
and 0.2390). The current revision broadens channel variation and predicts energy
fraction directly. Use `submission_v2.csv` and the V2 bundle; its leaderboard
performance has not yet been measured.

- [V2 execution instructions](solution/V2_README.md)
- [V2 failure analysis and full comparison](solution/V2_REPORT.md)

- [Historical V1 execution instructions](solution/README.md)
- [Historical V1 local evaluation](solution/VALIDATION_REPORT.md)
- [External-data declaration](solution/EXTERNAL_DATA_DISCLOSURE.md)
- [Experiment design](solution/EXPERIMENT_PLAN.md)

The solution combines spectral and temporal features, gradient-boosted models, and an energy-weighted neural network. Training and selection use released measurements and openly licensed recordings. Competition test inputs are used only for final inference.

Challenge data, audio downloads, model binaries and generated submissions are deliberately excluded from Git. The local submission bundle contains the trained weights needed for inference. No leaderboard rank is verified, and the external-source declaration must be posted as required by the challenge before submitting.
