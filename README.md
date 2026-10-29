# Underwater vessel-noise challenge

A reproducible local pipeline for estimating the controlled vessel-reference energy fraction in the South African underwater-noise challenge.

- [Execution instructions](solution/README.md)
- [Validation evidence and limitations](solution/VALIDATION_REPORT.md)
- [External-data declaration](solution/EXTERNAL_DATA_DISCLOSURE.md)
- [Experiment design](solution/EXPERIMENT_PLAN.md)

The solution combines spectral and temporal features, gradient-boosted models, and an energy-weighted neural network. Training and selection use released measurements and openly licensed recordings. Competition test inputs are used only for final inference.

Challenge data, audio downloads, model binaries and generated submissions are deliberately excluded from Git. The local submission bundle contains the trained weights needed for inference. No leaderboard rank is verified, and the external-source declaration must be posted as required by the challenge before submitting.
