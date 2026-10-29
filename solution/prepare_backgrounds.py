"""Build timestamp-respecting windows and frozen station/time holdouts."""

from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from audit_data import ROOT, COLS

DEST = ROOT / "solution" / "cache"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    arrays = []
    records = []
    rejected = []
    for path in sorted((ROOT / "South_African_Real_Acoustics/acoustics").glob("*.parquet")):
        df = pd.read_parquet(path, columns=["site", "recorder", "t", "dur"] + COLS)
        df["minute"] = pd.to_datetime(df.t).dt.as_unit("ns").astype("int64") // 60_000_000_000
        a = df[COLS].to_numpy(np.float32)
        bad = (
            (df.dur.to_numpy() < 55)
            | (df.dur.to_numpy() > 65)
            | (np.nanmax(a, 1) - np.nanmin(a, 1) < 6)
        )
        a[(a < 0) | (a > 220)] = np.nan
        a[bad] = np.nan
        df[COLS] = a
        site = int(df.site.iloc[0].split()[0])
        n_before = len(records)
        for recorder, g in df.groupby("recorder", sort=True):
            # Prefer the longest record when the same recorder-minute is duplicated.
            g = g.sort_values(["minute", "dur"]).drop_duplicates("minute", keep="last")
            minute = g.minute.to_numpy()
            block = minute // 15
            slot = minute % 15
            blocks, inverse = np.unique(block, return_inverse=True)
            windows = np.full((len(blocks), 15, 25), np.nan, np.float32)
            windows[inverse, slot] = g[COLS].to_numpy(np.float32)
            keep = np.isfinite(windows).sum((1, 2)) >= 125
            windows = windows[keep]
            blocks = blocks[keep]
            days = blocks // 96
            distinct = np.unique(days)
            if len(distinct) < 4:
                # Short deployment: split contiguous 6-hour blocks instead.
                groups = blocks // 24
                distinct = np.unique(groups)
            else:
                groups = days
            a_cut = max(1, int(len(distinct) * 0.70))
            b_cut = max(a_cut + 1, int(len(distinct) * 0.85))
            train_groups = distinct[:a_cut]
            val_groups = distinct[a_cut:b_cut]
            splits = np.where(
                np.isin(groups, train_groups),
                "train",
                np.where(np.isin(groups, val_groups), "val_time", "audit_time"),
            )
            if site == 2:
                splits[:] = "val_site"
            if site == 13:
                splits[:] = "audit_site"
            arrays.append(windows)
            for j, b in enumerate(blocks):
                uid = hashlib.sha256(f"{site}|{recorder}|{b}".encode()).hexdigest()[:20]
                records.append(
                    {
                        "window_id": uid,
                        "site": site,
                        "recorder": recorder,
                        "start_export_clock": str(pd.Timestamp(int(b) * 15 * 60, unit="s")),
                        "time_block": int(b),
                        "split": splits[j],
                        "observed_cells": int(np.isfinite(windows[j]).sum()),
                    }
                )
        rejected.append(
            {
                "site": site,
                "source_rows": len(df),
                "screened_rows": int(bad.sum()),
                "windows_kept": len(records) - n_before,
            }
        )
        print(rejected[-1], flush=True)
    windows = np.concatenate(arrays)
    meta = pd.DataFrame(records)
    assert len(meta) == len(windows) and meta.window_id.is_unique
    np.save(DEST / "backgrounds.npy", windows)
    meta.to_csv(DEST / "background_windows.csv", index=False)
    (ROOT / "solution/reports/window_preparation.json").write_text(
        json.dumps(
            {
                "sites": rejected,
                "split_counts": meta.groupby(["split", "site"]).size().to_dict().__str__(),
                "shape": list(windows.shape),
                "sha256_windows": hashlib.sha256(
                    (DEST / "backgrounds.npy").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )
    print(meta.groupby(["split", "site"]).size().to_string(), flush=True)


if __name__ == "__main__":
    main()
