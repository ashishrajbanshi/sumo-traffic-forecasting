#!/usr/bin/env python3
"""
fix_hdf5_alignment.py — Trim traffic_jan2026.h5 to exactly 8640 rows.

Each day's SUMO simulation ran slightly past midnight, producing 291–293
intervals instead of 288. After concatenation this leaves:
  - 100 duplicate timestamps at day boundaries (overlap between day N's
    overflow and day N+1's start)
  - 5 extra timestamps dated 2026-01-31 (overflow from Jan 30)
  - Total: 8745 rows instead of the expected 8640

Fix:
  1. Deduplicate — keep first occurrence (earlier day's simulation wins)
  2. Reindex to the canonical 8640-step range (Jan 1 00:00 → Jan 30 23:55)
     which drops the Jan 31 overflow and any other out-of-range rows

All 4 signals (speed, volume, occupancy, flow) are fixed in one pass.
Output is written back to the same file.
"""

import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR  = os.path.normpath(os.path.join(BASE_DIR, ".."))
H5_PATH  = os.path.join(SIM_DIR, "data", "traffic", "traffic_jan2026.h5")

CANONICAL = pd.date_range("2026-01-01 00:00", periods=8640, freq="5min")


def fix(h5_path):
    print(f"Input : {h5_path}")

    with pd.HDFStore(h5_path, "r") as store:
        signals = {key.lstrip("/"): store[key] for key in store.keys()}

    print(f"Signals found: {list(signals.keys())}")
    print(f"Canonical range: {CANONICAL[0]}  →  {CANONICAL[-1]}  ({len(CANONICAL)} steps)")
    print()

    fixed = {}
    for sig, df in signals.items():
        before = len(df)
        dups   = df.index.duplicated().sum()

        df = df[~df.index.duplicated(keep="first")]
        df = df.reindex(CANONICAL)

        missing = df.isna().all(axis=1).sum()
        if missing > 0:
            print(f"  WARNING /{sig}: {missing} rows are all-NaN after reindex — "
                  f"filling forward")
            df = df.ffill()

        fixed[sig] = df
        print(f"  /{sig:<10}  {before} → {len(df)} rows  "
              f"(removed {dups} duplicates, "
              f"dropped {before - dups - len(df)} overflow rows)")

    print(f"\nWriting fixed HDF5 → {h5_path}")
    with pd.HDFStore(h5_path, mode="w", complevel=5, complib="blosc") as store:
        for sig, df in fixed.items():
            store.put(sig, df)

    size_mb = os.path.getsize(h5_path) / 1e6
    print(f"Done — {size_mb:.1f} MB")

    # Final verification
    print("\nVerification:")
    with pd.HDFStore(h5_path, "r") as store:
        for key in store.keys():
            df = store[key]
            assert len(df) == 8640, f"{key} has {len(df)} rows, expected 8640"
            assert df.index.duplicated().sum() == 0, f"{key} still has duplicates"
            print(f"  {key:<12} shape={df.shape}  "
                  f"range={df.index[0].date()} → {df.index[-1].date()}  ✓")


if __name__ == "__main__":
    fix(H5_PATH)
