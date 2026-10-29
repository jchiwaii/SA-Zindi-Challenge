"""Reproducible controlled additions from permitted training-only sources."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from acoustic_model import energy

BASE = Path(__file__).resolve().parent


class Simulator:
    def __init__(self, background_splits=("train",), reference_splits=("train",)):
        self.backgrounds = np.load(BASE / "cache/backgrounds.npy", mmap_mode="r")
        self.meta = pd.read_csv(BASE / "cache/background_windows.csv")
        self.allowed = self.meta.index[self.meta.split.isin(background_splits)].to_numpy()
        sites = self.meta.loc[self.allowed, "site"].to_numpy()
        # Square-root site balancing retains coverage without letting Richards Bay dominate.
        unique, counts = np.unique(sites, return_counts=True)
        weights = {s: c**-0.5 for s, c in zip(unique, counts)}
        self.weights = np.asarray([weights[s] for s in sites])
        self.weights /= self.weights.sum()
        manifest = BASE / "cache/reference_manifest.csv"
        if manifest.exists():
            refs = pd.read_csv(manifest)
        else:
            refs = pd.DataFrame(
                [
                    json.loads(p.read_text())
                    for p in sorted((BASE / "external/vtuad_1.3.2/minute_bands").glob("*.json"))
                ]
            )
        self.refs = refs[refs.split.isin(reference_splits)].reset_index(drop=True)
        if len(self.refs) == 0:
            raise ValueError("No external references available")
        self.reference_arrays = [np.load(BASE / p) for p in self.refs.feature_path]
        count = self.refs.label.value_counts()
        self.reference_weights = self.refs.label.map(lambda l: count[l] ** -0.5).to_numpy()
        self.reference_weights /= self.reference_weights.sum()

    def generate(self, n, seed, scenario="standard", augment_background=True, return_cells=False):
        rng = np.random.default_rng(seed)
        idx = rng.choice(self.allowed, n, p=self.weights)
        b = np.array(self.backgrounds[idx], copy=True)
        ref_idx = rng.choice(len(self.refs), n, p=self.reference_weights)
        source = np.empty_like(b)
        xf = np.linspace(-1, 1, 25)
        time_axis = np.arange(15)
        for i, ri in enumerate(ref_idx):
            reference = self.reference_arrays[ri]
            length = len(reference)
            start = rng.uniform(0, max(length - 1, 0))
            span = rng.uniform(0.5, 1.5) * max(length - 1, 1)
            positions = np.linspace(start - span / 2, start + span / 2, 15)
            if rng.random() < 0.5:
                positions = positions[::-1]
            shape = np.stack(
                [np.interp(positions, np.arange(length), reference[:, f]) for f in range(25)],
                axis=1,
            )
            # Frequency-dependent propagation / recorder response; no absolute calibration assumed.
            scale = 2.8 if scenario != "channel_shift" else 6.0
            response = np.interp(xf, np.linspace(-1, 1, 6), rng.normal(0, scale, 6))
            response += rng.normal(0, 3) * xf + rng.normal(0, 1.5) * xf**2
            response -= rng.uniform(0, 8) * ((xf + 1) / 2) ** 3
            shape += response
            kind = rng.choice(4, p=[0.20, 0.40, 0.25, 0.15])
            center = rng.uniform(-2, 16)
            width = rng.uniform(1.2, 7)
            if scenario == "flat":
                kind = 0
            if kind == 0:
                env = np.ones(15)
            elif kind == 1:
                env = np.exp(-0.5 * ((time_axis - center) / width) ** 2)
            elif kind == 2:
                env = 1 / (1 + ((time_axis - center) / width) ** 2)
            else:
                env = np.exp(-0.5 * ((time_axis - center) / width) ** 2) + rng.uniform(
                    0.2, 1
                ) * np.exp(-0.5 * ((time_axis - rng.uniform(0, 14)) / rng.uniform(1, 4)) ** 2)
            floor = rng.uniform(0.01, 0.25)
            env = floor + (1 - floor) * env / max(env.max(), 1e-8)
            shape += (
                10 * np.log10(env)[:, None]
                + gaussian_filter1d(rng.normal(0, 0.6, 15), 0.7)[:, None]
            )
            source[i] = shape
        if augment_background:
            # Modest, independently drawn calibration perturbation, preserving original profiles.
            gain = rng.normal(0, 1.5, (n, 1, 1)).astype(np.float32)
            b += gain
        mask = np.isfinite(b)
        bg, offset = energy(b)
        src, _ = energy(source)
        if scenario == "weak":
            snr = rng.uniform(-30, -7, n)
        elif scenario == "strong":
            snr = rng.uniform(-5, 8, n)
        else:
            snr = rng.uniform(-30, 0, n)
            strong = rng.random(n) < 0.2
            snr[strong] = rng.uniform(-5, 8, strong.sum())
        is_added = rng.random(n) >= 0.25
        ratio = np.power(10.0, snr / 10) * is_added
        src_sum = (src * mask).sum((1, 2))
        src *= ((bg.sum((1, 2)) / (src_sum + 1e-25)) * ratio)[:, None, None]
        # Missingness is sampled independently. Recompute the target on remaining cells.
        missing_rate = 0.65 if scenario == "missing" else 0.30
        for i in np.flatnonzero(rng.random(n) < missing_rate):
            miss = rng.uniform(0.05, 0.4 if scenario != "missing" else 0.65)
            mask[i, rng.random(15) < miss, :] = False
            if rng.random() < 0.25:
                lo = rng.integers(0, 25)
                mask[i, :, lo : min(lo + rng.integers(1, 7), 25)] = False
        for i in np.flatnonzero(mask.sum((1, 2)) < 25):
            mask[i] = np.isfinite(b[i])
        mix = bg + src
        target = (src * mask).sum((1, 2)) / np.maximum((mix * mask).sum((1, 2)), 1e-25)
        result = np.where(mask, 10 * np.log10(np.maximum(mix, 1e-25)) + offset, np.nan).astype(
            np.float32
        )
        details = pd.DataFrame(
            {
                "background_index": idx,
                "window_id": self.meta.loc[idx, "window_id"].to_numpy(),
                "site": self.meta.loc[idx, "site"].to_numpy(),
                "background_split": self.meta.loc[idx, "split"].to_numpy(),
                "reference_hull": self.refs.loc[ref_idx, "hull_id"].to_numpy(),
                "reference_label": self.refs.loc[ref_idx, "label"].to_numpy(),
                "reference_member": self.refs.loc[ref_idx, "archive_member"].to_numpy(),
                "scenario": scenario,
                "seed": seed,
                "is_added": is_added,
                "share": target,
                "observed_cells": mask.sum((1, 2)),
            }
        )
        if return_cells:
            cells = np.where(mask, src / np.maximum(mix, 1e-25), 0).astype(np.float32)
            return result, target.astype(np.float32), details, cells
        return result, target.astype(np.float32), details
