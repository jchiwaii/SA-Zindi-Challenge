# Vessel addition energy estimation

Plan written before loading any test acoustic values or running model training.

## Objective and guardrails

Predict the controlled added-reference energy fraction. Natural vessel presence is a different, unlabelled target. All training and selection use released training measurements and public licensed data. Competition test inputs are read only by the final inference command; their values, distributions, IDs and possible source identities do not inform training, augmentation, calibration, model selection, or preprocessing estimates. No leaderboard probes or AutoML.

## Evidence gathered

The supplied starter explicitly attributes the passage-reference corpus to VTUAD v1.3.2, CC BY 4.0. Its public strict-interval recordings avoid known archive-container extensions. We use these as a generic acoustic training corpus, not as a dictionary to identify or reconstruct competition test sources. Keep hulls/recordings separate during validation. Public South African measurements contain duration issues, sentinel values, strongly varying spectral response, gaps and repeated measurements. Absolute pressure calibration is unverified.

## Planned approach

1. Audit the training release and construct 15-minute windows within recorder and contiguous time blocks. Reject clearly invalid rows, preserve missing cells, and retain window identifiers.
2. Convert public WAV recordings into minute-band energy using the supplied starter's Hann-periodogram convention. Preserve source attribution, hashes, exact version and processing settings.
3. Hold out complete South African periods and stations; separately hold out external hulls. Do not let augmented versions of the same background cross a split.
4. Generate supervised examples by adding external reference energy to real backgrounds in **linear energy**, with varied temporal envelopes, channel responses, gain, and missingness. The measured-cell mask is shared by numerator and denominator. Include no-addition controls without calling them vessel-free.
5. Compare a constant, received-energy baseline, physically motivated temporal/spectral features, and manually specified gradient-boosted regression models. Evaluate share fraction RMSE and diagnostic AUC. Consider a compact neural model if warranted by held-out results and local resource availability.
6. Stress-test unknown addition strength, source channels, masking, temporal structure and unseen sites. Report sensitivity to augmentation assumptions; local synthetic scores are not estimates of leaderboard rank.
7. Freeze the chosen models and settings, run final test inference, verify all 1,500 IDs and bounds, and reproduce the submission from saved artifacts.

## Deliverables

Submission CSV, trained models, standalone reproducible Python scripts, dependency versions, split manifests, data/source audit, validation report, run-time/memory record, and a draft external-data disclosure for the public discussion board. Posting to an account and submitting to Zindi require the user's explicit instruction.
