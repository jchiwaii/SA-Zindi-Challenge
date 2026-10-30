# External acoustic data declaration — draft for the Zindi discussion board

This solution uses the public **VTUAD: An AIS-Auto-Labeled Vessel-Type Underwater Acoustic Dataset Reconstructed from Ocean Networks Canada Archives**, version **1.3.2**, by Hao Yuan, Wenbo Wang, Tian Li, Guici Chen, Lingjiang Zeng and Sixian Liu.

- Versioned source: https://doi.org/10.5281/zenodo.23037895
- Direct record: https://zenodo.org/records/23037895
- Machine-readable file listing: https://zenodo.org/api/records/23037895
- Licence: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/
- Upstream recordings: Ocean Networks Canada, Oceans 3.0, https://data.oceannetworks.ca/ ; repository identifier https://doi.org/10.17616/R3RW43
- Access: direct public downloads, without a subscription, payment card or API token. Download the three `vtuad_strict_wav_vessel.zip.part01`–`part03` files and concatenate in order, following the provider's `README_vtuad_strict_wav_vessel_parts.txt`. Verify the archive and per-WAV checksums supplied by the provider. An HTTP byte-range downloader can alternatively fetch the same selected compressed members and check their original WAV SHA-256 hashes.

The selected subset is deterministic: choose the smallest compressed strict-interval recording for each pseudonymous hull, add up to eight smallest recordings per vessel class, and remove duplicate recording selections. This yields 201 recordings spanning 172 hulls. The selection uses public training-corpus metadata only. Exact selections and SHA-256 hashes accompany the reproducibility files. Hull identifiers are used only to isolate validation splits, never as model inputs or to identify competition test sources.

Audio preprocessing: use the first waveform channel and the file's actual sample rate; convert integer PCM to floating point; divide into non-overlapping two-second segments; remove the segment mean; calculate Hann-window one-sided power spectral density; integrate into 25 base-10 third-octave bands with centres `10 ** (1.5 + 0.1 * arange(25))`; average 30 segment powers in linear units for each complete minute; omit a trailing incomplete minute. Discard absolute digital gain. These are relative acoustic energies, not calibrated pressure. The per-file conversion is a reimplementation of the optional conversion shown in the competition starter notebook.

Permitted training augmentations combine these reference patterns with released South African background measurements in linear energy, with varied passage timing, relative frequency response, strength and observation masks. Targets use exactly the same observed cells in their added-energy numerator and total-energy denominator. Reference recordings can contain ambient sound; AIS-derived vessel categories are not manually verified acoustic labels.

No competition test measurements are used for training, fitting preprocessing, tuning, pseudo-labelling or calibrating predictions. No source-identification matching or leaderboard label reconstruction is performed. The South African data remain subject to the competition's own data-use restrictions; this declaration does not apply CC BY 4.0 to those measurements.

This file is a draft, **not a record that the disclosure has been posted**. Publish the source/version/access information in the competition discussion board before relying on external-data eligibility for a submission.

Revision 2 retains exactly the same external corpus and subset. Its training
augmentation additionally varies broad spectral tilt and curvature, transfers
relative reference profiles toward independently sampled released South African
spectra, and applies smooth relative channel responses. Reference temporal
variation remains recorded data. Background dither and export rounding are
independent of the addition label. These are declared robustness augmentations,
not assertions that a particular channel model matches the hidden test generator.
