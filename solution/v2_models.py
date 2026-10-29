"""Direct regression and fixed temporal features for the transfer study."""

import warnings
import numpy as np
from acoustic_model import energy, features


def build_features(db):
    original, names = features(db)
    extras = []
    for start in range(0, len(db), 1024):
        x = db[start : start + 1024]
        e, _ = energy(x)
        mask = np.isfinite(x)
        count = mask.sum(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            observed = np.where(mask, e, np.nan)
            baseline = np.nanpercentile(observed, 20, axis=1)
            cv = np.nanstd(observed, axis=1) / np.maximum(np.nanmean(observed, axis=1), 1e-20)
        total = e.sum(1)
        excess = np.maximum(total - baseline * count, 0) / np.maximum(total, 1e-20)
        weights = total / np.maximum(total.sum(1, keepdims=True), 1e-20)
        cv = np.nan_to_num(cv, nan=0.0)
        excess = np.nan_to_num(excess, nan=0.0)
        pooled = np.column_stack(
            [
                (cv * weights).sum(1),
                (excess * weights).sum(1),
                (cv**2 * weights).sum(1),
                (excess**2 * weights).sum(1),
            ]
        )
        extras.append(np.column_stack([cv, excess, pooled]))
    names += [f"linear_cv_{i}" for i in range(25)]
    names += [f"linear_excess_{i}" for i in range(25)]
    names += ["weighted_cv", "weighted_excess", "weighted_cv2", "weighted_excess2"]
    values = np.column_stack([original, np.concatenate(extras)]).astype(np.float32)
    invariant = []
    for i, name in enumerate(names):
        if name.startswith(
            (
                "band_std",
                "band_range",
                "band_iqr",
                "time_",
                "temporal_",
                "coverage",
                "band_coverage",
                "minute_coverage",
                "linear_",
                "weighted_",
            )
        ):
            invariant.append(i)
        elif name.startswith("group") and not name.endswith(("_db", "_share")):
            invariant.append(i)
    return values, names, np.asarray(invariant)
