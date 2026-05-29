#!/usr/bin/env python3
"""
prepare_data.py — Adapt sumo_3 data into DCRNN-compatible format.

Reads from:
  ../data/graph/node_ids.txt
  ../data/graph/edges.csv
  ../data/traffic/traffic_jan2026.h5  (key: /speed)

Writes to:
  data/sensor_graph/node_ids.txt           — filtered node IDs (bad nodes removed)
  data/sensor_graph/distances_chattanooga.csv — edges with DCRNN column names
  data/traffic_chattanooga.h5              — single-key speed HDF5

Run from sumo_3/DCRNN/:
    python scripts/prepare_data.py
"""

import os
import sys
import pandas as pd

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DCRNN_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))
SUMO_DIR  = os.path.normpath(os.path.join(DCRNN_DIR, ".."))

# ── Input paths ────────────────────────────────────────────────────────────────
NODE_IDS_IN  = os.path.join(SUMO_DIR, "data", "graph", "node_ids.txt")
EDGES_IN     = os.path.join(SUMO_DIR, "data", "graph", "edges.csv")
TRAFFIC_IN   = os.path.join(SUMO_DIR, "data", "traffic", "traffic_jan2026.h5")

# ── Output paths ───────────────────────────────────────────────────────────────
SENSOR_GRAPH_DIR = os.path.join(DCRNN_DIR, "data", "sensor_graph")
NODE_IDS_OUT     = os.path.join(SENSOR_GRAPH_DIR, "node_ids.txt")
DISTANCES_OUT    = os.path.join(SENSOR_GRAPH_DIR, "distances_chattanooga.csv")
TRAFFIC_OUT      = os.path.join(DCRNN_DIR, "data", "traffic_chattanooga.h5")

# ── Nodes to exclude ───────────────────────────────────────────────────────────
# 2 isolated nodes (no reachable neighbors within 2000m in either direction)
ISOLATED_NODES = {"19509587", "39452650#1"}

# 7 zero-volume nodes (no vehicles observed across all 30 days — speed is
# entirely speed-limit fill, not real measurements)
ZERO_VOLUME_NODES = {
    "-1155817285#1",
    "-1156995281",
    "-19501516#19",
    "-758017387#5",
    "174645263",
    "51067022#0",
    "686208415",
}

BAD_NODES = ISOLATED_NODES | ZERO_VOLUME_NODES


def main():
    os.makedirs(SENSOR_GRAPH_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(TRAFFIC_OUT), exist_ok=True)

    # ── 1. Filter node IDs ─────────────────────────────────────────────────────
    print("Reading node IDs ...")
    with open(NODE_IDS_IN) as f:
        all_ids = [x for x in f.read().strip().split(",") if x]

    clean_ids = [n for n in all_ids if n not in BAD_NODES]
    removed   = len(all_ids) - len(clean_ids)

    print(f"  Total nodes   : {len(all_ids)}")
    print(f"  Removed (bad) : {removed}  {sorted(BAD_NODES)}")
    print(f"  Clean nodes   : {len(clean_ids)}")

    with open(NODE_IDS_OUT, "w") as f:
        f.write(",".join(clean_ids))
    print(f"  Saved → {NODE_IDS_OUT}")

    clean_set = set(clean_ids)

    # ── 2. Adapt edges.csv → distances_chattanooga.csv ────────────────────────
    print("\nAdapting edges.csv ...")
    edges = pd.read_csv(EDGES_IN, dtype={"from_node": str, "to_node": str})

    # Keep only edges where both endpoints are clean nodes
    mask  = edges["from_node"].isin(clean_set) & edges["to_node"].isin(clean_set)
    edges = edges[mask].copy()

    # Rename to DCRNN expected column names
    edges = edges.rename(columns={
        "from_node":  "from",
        "to_node":    "to",
        "distance_m": "cost",
    })

    edges.to_csv(DISTANCES_OUT, index=False)
    print(f"  Edges before filter : {mask.shape[0]:,}")
    print(f"  Edges after filter  : {len(edges):,}")
    print(f"  Saved → {DISTANCES_OUT}")

    # ── 3. Extract /speed and write single-key HDF5 ───────────────────────────
    print("\nExtracting speed signal from HDF5 ...")
    store = pd.HDFStore(TRAFFIC_IN, "r")
    speed = store["speed"]
    store.close()

    # Keep only clean node columns
    cols_to_keep = [c for c in speed.columns if c in clean_set]
    speed = speed[cols_to_keep]

    print(f"  Shape : {speed.shape}  (timesteps × nodes)")
    print(f"  Range : {speed.index[0]}  →  {speed.index[-1]}")
    print(f"  Speed : min={speed.values.min():.1f}  "
          f"mean={speed.values.mean():.1f}  max={speed.values.max():.1f} mph")

    # Downcast index to datetime64[ns] so pandas 0.20.x (dcrnn env) can read it
    speed.index = speed.index.astype("datetime64[ns]")

    # Write as single-key HDF5 — pd.read_hdf() will read it directly
    speed.to_hdf(TRAFFIC_OUT, key="df", mode="w", complevel=5, complib="blosc")
    size_mb = os.path.getsize(TRAFFIC_OUT) / 1e6
    print(f"  Saved → {TRAFFIC_OUT}  ({size_mb:.1f} MB)")

    # ── 4. Summary ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Data preparation complete. Next steps:")
    print()
    print("  # 1. Build adjacency matrix")
    print("  python scripts/gen_adj_mx.py \\")
    print(f"      --sensor_ids_filename data/sensor_graph/node_ids.txt \\")
    print(f"      --distances_filename  data/sensor_graph/distances_chattanooga.csv \\")
    print(f"      --output_pkl_filename data/sensor_graph/adj_mx_chattanooga.pkl")
    print()
    print("  # 2. Generate sliding-window train/val/test splits")
    print("  python scripts/generate_training_data.py \\")
    print(f"      --traffic_df_filename data/traffic_chattanooga.h5 \\")
    print(f"      --output_dir          data/chattanooga/")
    print()
    print("  # 3. Train DCRNN")
    print("  python dcrnn_train.py \\")
    print(f"      --config_filename data/model/dcrnn_chattanooga.yaml")
    print("="*60)


if __name__ == "__main__":
    main()
