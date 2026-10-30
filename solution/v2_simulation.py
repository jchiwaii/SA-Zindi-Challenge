"""Domain-randomized additions; no test-input access or source matching."""

import warnings
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from acoustic_model import energy
from simulation import Simulator

FAMILIES = ("native", "tilted", "local_channel", "smooth_channel")


class DiverseSimulator(Simulator):
    def generate(
        self, n, seed, family="mixed", strength="mixed", return_cells=False, paired_channels=False
    ):
        rng = np.random.default_rng(seed)
        idx = rng.choice(self.allowed, n, p=self.weights)
        other = rng.choice(self.allowed, n, p=self.weights)
        background = np.array(self.backgrounds[idx], copy=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            profiles = np.nanmedian(self.backgrounds[other], axis=1)
            profiles -= np.nanmean(profiles, axis=1, keepdims=True)
        # Recorder frequency coverage can differ: interpolate the response profile.
        for i in range(n):
            valid = np.isfinite(profiles[i])
            profiles[i] = np.interp(np.arange(25), np.flatnonzero(valid), profiles[i, valid])
        reference_indices = rng.choice(len(self.refs), n, p=self.reference_weights)
        families = rng.choice(FAMILIES, n) if family == "mixed" else np.full(n, family)
        source = np.empty_like(background)
        coordinate = np.linspace(-1, 1, 25)
        t = np.arange(15)
        for i, ri in enumerate(reference_indices):
            reference = self.reference_arrays[ri]
            length = len(reference)
            positions = np.linspace(rng.uniform(-1, 1), length - 1 + rng.uniform(-1, 1), 15)
            if rng.random() < 0.5:
                positions = positions[::-1]
            original = np.stack(
                [np.interp(positions, np.arange(length), reference[:, j]) for j in range(25)],
                axis=1,
            )
            mean = original.mean(0)
            residual = original - mean
            channel_rng = (
                np.random.default_rng(np.random.SeedSequence([seed, i, 2107]))
                if paired_channels
                else rng
            )
            kind = families[i]
            if kind == "native":
                shape = mean.copy()
                response_scale = 3.0
            elif kind == "tilted":
                shape = mean + channel_rng.uniform(-24, 12) * coordinate
                shape += channel_rng.uniform(-10, 10) * coordinate**2
                response_scale = 5.0
            elif kind == "local_channel":
                mix = channel_rng.uniform(0.65, 1.0)
                shape = mix * profiles[i] + (1 - mix) * mean
                shape += channel_rng.uniform(-8, 8) * coordinate
                response_scale = 4.0
            elif kind == "smooth_channel":
                shape = (
                    channel_rng.uniform(-28, 12) * coordinate
                    + channel_rng.uniform(-12, 12) * coordinate**2
                )
                response_scale = 7.0
            elif kind == "held_channel":
                # A distinct smooth response construction, absent from fitting.
                shape = profiles[i] + channel_rng.uniform(3, 12) * np.sin(
                    channel_rng.uniform(0.5, 2) * np.pi * coordinate + channel_rng.uniform(-3, 3)
                )
                response_scale = 1.5
            else:
                raise ValueError(kind)
            response = np.interp(
                coordinate, np.linspace(-1, 1, 5), channel_rng.normal(0, response_scale, 5)
            )
            residual *= rng.uniform(0.3, 1.8)
            envelope_kind = rng.integers(4)
            center, width = rng.uniform(-4, 19), rng.uniform(0.7, 10)
            if envelope_kind == 0:
                envelope = np.ones(15)
            elif envelope_kind == 1:
                envelope = np.exp(-0.5 * ((t - center) / width) ** 2)
            elif envelope_kind == 2:
                envelope = 1 / (1 + ((t - center) / width) ** 2)
            else:
                envelope = np.exp(gaussian_filter1d(rng.normal(0, 1.2, 15), 1))
            envelope /= envelope.max()
            floor = rng.uniform(0.001, 0.25)
            source[i] = (
                shape + response + residual + 10 * np.log10(floor + (1 - floor) * envelope)[:, None]
            )
        # Dither is independent of presence, eliminating exact decimal-grid cues.
        background += rng.uniform(-0.05, 0.05, background.shape).astype(np.float32)
        background += rng.normal(0, 2.5, (n, 1, 1)).astype(np.float32)
        observed = np.isfinite(background)
        for i in np.flatnonzero(rng.random(n) < 0.3):
            missing = rng.uniform(0.05, 0.4)
            proposed = observed[i].copy()
            proposed[rng.random(15) < missing, :] = False
            if proposed.sum() >= 25:
                observed[i] = proposed
        bg, offset = energy(background)
        src, _ = energy(source)
        regimes = (
            rng.choice(["weak", "broad", "strong"], n, p=[0.4, 0.4, 0.2])
            if strength == "mixed"
            else np.full(n, strength)
        )
        desired = np.zeros(n)
        for regime in np.unique(regimes):
            selected = regimes == regime
            count = selected.sum()
            if regime == "weak":
                ratio = 10 ** (rng.uniform(-30, -7, count) / 10)
                desired[selected] = ratio / (1 + ratio)
            elif regime == "broad":
                desired[selected] = rng.uniform(0.01, 0.65, count)
            elif regime == "strong":
                desired[selected] = rng.uniform(0.25, 0.85, count)
            else:
                raise ValueError(regime)
        present = rng.random(n) >= 0.25
        desired *= present
        ratio = desired / np.maximum(1 - desired, 1e-8)
        src *= (((bg * observed).sum((1, 2)) / (src * observed).sum((1, 2))) * ratio)[:, None, None]
        mixture = bg + src
        target = (src * observed).sum((1, 2)) / (mixture * observed).sum((1, 2))
        db = np.where(observed, 10 * np.log10(np.maximum(mixture, 1e-25)) + offset, np.nan)
        # Observation error is independent of presence; target is the latent mixture.
        step = rng.choice([0.001, 0.01, 0.1], n, p=[0.15, 0.35, 0.50])[:, None, None]
        db = (np.round(db / step) * step).astype(np.float32)
        meta = pd.DataFrame(
            {
                "background_index": idx,
                "window_id": self.meta.loc[idx, "window_id"].to_numpy(),
                "site": self.meta.loc[idx, "site"].to_numpy(),
                "reference_hull": self.refs.loc[reference_indices, "hull_id"].to_numpy(),
                "reference_member": self.refs.loc[reference_indices, "archive_member"].to_numpy(),
                "family": families,
                "strength": regimes,
                "is_added": present,
                "share": target,
            }
        )
        if return_cells:
            cells = np.where(observed, src / np.maximum(mixture, 1e-25), 0).astype(np.float32)
            return db, target.astype(np.float32), meta, cells
        return db, target.astype(np.float32), meta
