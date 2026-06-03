#!/usr/bin/env python3
"""
parse_detector_output.py — Convert SUMO E1 detector XML into a multi-signal HDF5.

Two modes:

  --jan2026-dir output/jan2026          (PRIMARY — January 2026 real-date runs)
      Reads all 30 YYYY-MM-DD subdirectories, uses the actual calendar date
      from the directory name as the timestamp anchor, and aggregates
      multi-lane detectors on the same edge into a single node.
      Output columns = edge IDs (graph nodes).

  --day-type weekday|weekend            (LEGACY — synthetic cycling)
      Original mode for manually named detector output files.
      Output columns = individual detector IDs.

Output (HDF5 keys — all shape: timesteps × nodes)
──────
  /speed      mean speed across lanes, mph; no-vehicle intervals → speed limit
  /volume     total vehicles entered across lanes per 5-min interval
  /occupancy  mean occupancy across lanes, %; no-vehicle intervals → 0.0
  /flow       mean flow across lanes, veh/hour; no-vehicle intervals → 0.0

Usage
─────
    # January 2026 — all 30 days, real dates, lane-aggregated nodes
    python3 script/parse_detector_output.py --jan2026-dir output/jan2026

    # Legacy single run
    python3 script/parse_detector_output.py --day-type weekday
"""

import os
import sys
import argparse
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR  = os.path.normpath(os.path.join(BASE_DIR, ".."))

DAY_CONFIG = {
    "weekday": {
        "det_output":   os.path.join(SIM_DIR, "detectors", "detector_output_weekday.xml"),
        "filtered_add": os.path.join(SIM_DIR, "detectors", "detectors_filtered_weekday.add.xml"),
        "all_add":      os.path.join(SIM_DIR, "detectors", "detectors_weekday.add.xml"),
        "out_h5":       os.path.join(SIM_DIR, "data",      "traffic", "traffic_weekday.h5"),
        "day_offsets":  [0, 1, 2, 3, 4],
        "base_anchor":  "2026-05-18",
        "week_advance": 7,
    },
    "weekend": {
        "det_output":   os.path.join(SIM_DIR, "detectors", "detector_output_weekend.xml"),
        "filtered_add": os.path.join(SIM_DIR, "detectors", "detectors_filtered_weekend.add.xml"),
        "all_add":      os.path.join(SIM_DIR, "detectors", "detectors_weekend.add.xml"),
        "out_h5":       os.path.join(SIM_DIR, "data",      "traffic", "traffic_weekend.h5"),
        "day_offsets":  [0, 1],
        "base_anchor":  "2026-05-16",
        "week_advance": 7,
    },
}

MS_TO_MPH  = 2.23694
INTERVAL_S = 300

# signal → (xml_attr, fill_when_no_vehicles, lane_aggregation)
# fill "ref_speed" is special — replaced with the per-detector speed-limit value
SIGNALS = {
    "speed":     ("speed",     "ref_speed", "mean"),
    "volume":    ("entered",   0.0,         "sum"),   # total vehicles through section
    "occupancy": ("occupancy", 0.0,         "mean"),  # % of time lane occupied
    "flow":      ("flow",      0.0,         "mean"),  # veh/hour per lane
}


# ── Sumolib loader ─────────────────────────────────────────────────────────────

def _load_sumolib():
    for path in [None, os.path.join(os.environ["SUMO_HOME"], "tools"),
                 os.path.join(os.environ.get("SUMO_HOME", ""), "tools")]:
        if path:
            sys.path.insert(0, path)
        try:
            import sumolib
            return sumolib
        except ImportError:
            pass
    print("ERROR: sumolib not found.")
    sys.exit(1)


# ── Detector definition reader ─────────────────────────────────────────────────

def read_detector_defs(add_xml_path):
    """Returns {det_id: lane_id}."""
    mapping = {}
    for _, elem in ET.iterparse(add_xml_path, events=("end",)):
        if elem.tag == "e1Detector":
            det_id  = elem.get("id")
            lane_id = elem.get("lane")
            if det_id and lane_id:
                mapping[det_id] = lane_id
            elem.clear()
    return mapping


def _best_add_file(filtered_path, fallback_path):
    if os.path.exists(filtered_path):
        return filtered_path
    print(f"  Note: filtered file not found, using unfiltered:\n    {fallback_path}")
    return fallback_path


# ── Lane → edge ────────────────────────────────────────────────────────────────

def lane_to_edge(lane_id):
    return lane_id.rsplit("_", 1)[0]


# ── Reference speed cache ──────────────────────────────────────────────────────

def build_ref_speeds(det_to_lane, net_file):
    sumolib = _load_sumolib()
    print(f"  Loading network for reference speeds …")
    net = sumolib.net.readNet(net_file, withInternal=False)
    ref = {}
    missing = 0
    for det_id, lane_id in det_to_lane.items():
        try:
            edge_id  = lane_id.rsplit("_", 1)[0]
            lane_idx = int(lane_id.rsplit("_", 1)[1])
            ref[det_id] = net.getEdge(edge_id).getLane(lane_idx).getSpeed() * MS_TO_MPH
        except Exception:
            ref[det_id] = 30.0
            missing += 1
    if missing:
        print(f"  Warning: {missing} lanes not found — defaulted to 30 mph")
    return ref


# ── XML interval parser ────────────────────────────────────────────────────────

def parse_intervals(xml_path, keep_ids=None):
    """
    Returns:
        intervals : {det_id: {begin_s: {"speed": v, "volume": v,
                                         "occupancy": v, "flow": v}}}
        step_set  : sorted list of begin times

    Float fields absent when sampledSeconds=0 are stored as -1 (sentinel for
    "no vehicles") and handled per-signal in build_dataframe.
    """
    intervals = defaultdict(dict)
    step_set  = set()
    total     = 0

    print(f"  Parsing {os.path.basename(xml_path)} …", end="", flush=True)
    for _, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag != "interval":
            elem.clear()
            continue
        det_id = elem.get("id")
        begin  = float(elem.get("begin", 0))

        if keep_ids is not None and det_id not in keep_ids:
            elem.clear()
            continue

        intervals[det_id][begin] = {
            "speed":     float(elem.get("speed",     -1)),
            "volume":    float(elem.get("entered",    0)),   # always present
            "occupancy": float(elem.get("occupancy", -1)),
            "flow":      float(elem.get("flow",      -1)),
        }
        elem.clear()
        step_set.add(begin)
        total += 1

    print(f" {total:,} intervals, {len(intervals):,} detectors, {len(step_set)} timesteps")
    return dict(intervals), sorted(step_set)


# ── DataFrame builder (lane level) ────────────────────────────────────────────

def build_dataframe(intervals, step_set, ref_speeds, anchor_date, det_ids,
                    signal, fill_value):
    """
    Build a (timesteps × detectors) DataFrame for one signal.

    fill_value: float to use when a reading is absent (-1 sentinel),
                or the string "ref_speed" to use the per-detector speed limit.
    """
    n_steps      = len(step_set)
    begin_to_row = {b: i for i, b in enumerate(step_set)}
    use_ref      = (fill_value == "ref_speed")

    data = {}
    for det_id in det_ids:
        ref = ref_speeds.get(det_id, 30.0) if use_ref else fill_value
        col = [ref] * n_steps

        for begin, vals in intervals.get(det_id, {}).items():
            row = begin_to_row.get(begin)
            if row is None:
                continue
            raw = vals[signal]
            if raw < 0.0:
                col[row] = ref          # absent reading → fill
            elif signal == "speed":
                col[row] = raw * MS_TO_MPH
            else:
                col[row] = raw

        data[det_id] = col

    idx = pd.date_range(start=anchor_date, periods=n_steps, freq=f"{INTERVAL_S}s")
    df  = pd.DataFrame(data, index=idx)
    df.index.name = "timestamp"
    return df


# ── Lane → node aggregation ────────────────────────────────────────────────────

def aggregate_lanes_to_nodes(df, agg="mean", det_to_edge=None):
    """
    Collapse per-lane detector columns into per-edge (node) columns.

    agg="mean"    speed, occupancy, flow  — averaged across lanes
    agg="sum"     volume                  — summed across lanes (total vehicles)
    det_to_edge   {det_id: original_edge_id} — when provided, uses the original
                  SUMO edge IDs (e.g. '-1044300151#1') so column names match
                  nodes.csv / node_ids.txt / edges.csv exactly.
                  Falls back to string-parsing of sanitised names if absent.
    """
    col_to_edge = {}
    for col in df.columns:
        if det_to_edge is not None and col in det_to_edge:
            col_to_edge[col] = det_to_edge[col]
        else:
            # legacy fallback: derive from sanitised detector name
            inner = col[4:] if col.startswith("det_") else col
            inner = re.sub(r"_\d{5}$", "", inner)
            col_to_edge[col] = inner.rsplit("_", 1)[0]

    edge_groups = defaultdict(list)
    for col, edge in col_to_edge.items():
        edge_groups[edge].append(col)

    node_data = {}
    for edge_id, cols in sorted(edge_groups.items()):
        node_data[edge_id] = (
            df[cols].sum(axis=1) if agg == "sum" else df[cols].mean(axis=1)
        )

    result = pd.DataFrame(node_data, index=df.index)
    result.index.name = "timestamp"
    return result


# ── January 2026 mode ──────────────────────────────────────────────────────────

def run_jan2026(input_dir, net_file, output_h5, aggregate):
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    entries = sorted([
        d for d in os.listdir(input_dir)
        if date_pattern.match(d) and os.path.isdir(os.path.join(input_dir, d))
    ])

    if not entries:
        print(f"ERROR: No YYYY-MM-DD directories found in {input_dir}")
        sys.exit(1)

    print(f"\n  Found {len(entries)} daily directories in {input_dir}")

    add_weekday = _best_add_file(
        os.path.join(SIM_DIR, "detectors", "detectors_filtered_weekday.add.xml"),
        os.path.join(SIM_DIR, "detectors", "detectors_weekday.add.xml"),
    )
    add_weekend = _best_add_file(
        os.path.join(SIM_DIR, "detectors", "detectors_filtered_weekend.add.xml"),
        os.path.join(SIM_DIR, "detectors", "detectors_weekend.add.xml"),
    )
    wd_defs     = read_detector_defs(add_weekday)
    we_defs     = read_detector_defs(add_weekend)
    all_defs    = {**we_defs, **wd_defs}
    all_det_ids = set(all_defs.keys())
    print(f"  Known detectors: {len(all_det_ids):,}")

    # det_id → original SUMO edge ID (used to name columns correctly)
    det_to_edge = {det_id: lane_to_edge(lane_id)
                   for det_id, lane_id in all_defs.items()}

    ref_speeds = build_ref_speeds(all_defs, net_file)

    # One list of day-frames per signal
    signal_frames = {sig: [] for sig in SIGNALS}

    for date_str in entries:
        anchor   = pd.Timestamp(date_str)
        suffix   = "weekend" if anchor.weekday() >= 5 else "weekday"
        xml_path = os.path.join(input_dir, date_str, f"detector_output_{suffix}.xml")
        if not os.path.exists(xml_path):
            print(f"  Skipping {date_str} — detector_output_{suffix}.xml not found")
            continue

        day_name = anchor.strftime("%A")
        print(f"\n  {date_str} {day_name}")

        intervals, step_set = parse_intervals(xml_path, keep_ids=all_det_ids)
        if not step_set:
            print(f"    Warning: no intervals — skipping")
            continue

        for sig, (attr, fill, agg_fn) in SIGNALS.items():
            df = build_dataframe(intervals, step_set, ref_speeds, anchor,
                                 sorted(all_det_ids), sig, fill)
            if aggregate:
                df = aggregate_lanes_to_nodes(df, agg=agg_fn,
                                              det_to_edge=det_to_edge)
            signal_frames[sig].append(df)

        # Report shape once (same for all signals)
        sample = signal_frames["speed"][-1]
        print(f"    Shape: {sample.shape}  ({sample.index[0]} → {sample.index[-1]})")
        if aggregate:
            print(f"    Nodes after lane aggregation: {sample.shape[1]}")

    if not any(signal_frames.values()):
        print("ERROR: No valid frames produced.")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_h5), exist_ok=True)
    print(f"\nWriting → {output_h5}")

    with pd.HDFStore(output_h5, mode="w", complevel=5, complib="blosc") as store:
        for sig in SIGNALS:
            frames = signal_frames[sig]
            if not frames:
                print(f"  /{sig:<10} — no data, skipped")
                continue
            combined = pd.concat(frames).sort_index()
            store.put(sig, combined)
            vals = combined.values
            print(f"  /{sig:<10} shape={combined.shape}  "
                  f"min={vals.min():.2f}  mean={vals.mean():.2f}  max={vals.max():.2f}")

    print(f"Done — {os.path.getsize(output_h5)/1e6:.1f} MB")


# ── Legacy mode ────────────────────────────────────────────────────────────────

def run_legacy(args):
    cfg      = DAY_CONFIG[args.day_type]
    net_file = args.net_file or os.path.join(SIM_DIR, "network", "osm.net.xml")
    out_h5   = args.output   or cfg["out_h5"]

    add_file = _best_add_file(cfg["filtered_add"], cfg["all_add"])
    if args.no_filter:
        add_file = cfg["all_add"]

    print(f"\nLegacy mode — parse_detector_output ({args.day_type})")
    print(f"  Detector definitions : {add_file}")
    det_to_lane = read_detector_defs(add_file)
    det_ids     = sorted(det_to_lane.keys())
    print(f"  Detectors: {len(det_ids):,}")

    det_to_edge = {det_id: lane_to_edge(lane_id)
                   for det_id, lane_id in det_to_lane.items()}

    ref_speeds = build_ref_speeds(det_to_lane, net_file)

    run_files = [cfg["det_output"]] + list(args.extra_runs or [])
    for f in run_files:
        if not os.path.exists(f):
            print(f"ERROR: detector output not found:\n  {f}")
            sys.exit(1)

    offsets     = cfg["day_offsets"]
    cycle_len   = len(offsets)
    base_anchor = pd.Timestamp(cfg["base_anchor"])
    keep        = set(det_ids)

    signal_frames = {sig: [] for sig in SIGNALS}

    for run_idx, xml_path in enumerate(run_files):
        week_num   = run_idx // cycle_len
        day_offset = offsets[run_idx % cycle_len] + week_num * cfg["week_advance"]
        run_anchor = base_anchor + pd.Timedelta(days=day_offset)
        day_name   = run_anchor.strftime("%A %Y-%m-%d")
        print(f"\nRun {run_idx+1}/{len(run_files)} [{day_name}]: {xml_path}")

        intervals, step_set = parse_intervals(xml_path, keep_ids=keep)
        if not step_set:
            print("  Warning: no intervals — skipping.")
            continue

        for sig, (attr, fill, agg_fn) in SIGNALS.items():
            df = build_dataframe(intervals, step_set, ref_speeds, run_anchor,
                                 det_ids, sig, fill)
            signal_frames[sig].append(df)
            # note: legacy mode does not aggregate lanes — columns stay as det IDs

        sample = signal_frames["speed"][-1]
        print(f"  Shape: {sample.shape}  ({sample.index[0]} → {sample.index[-1]})")

    if not any(signal_frames.values()):
        print("ERROR: No valid frames.")
        sys.exit(1)

    os.makedirs(os.path.dirname(out_h5), exist_ok=True)
    print(f"\nWriting → {out_h5}")

    with pd.HDFStore(out_h5, mode="w", complevel=5, complib="blosc") as store:
        for sig in SIGNALS:
            frames = signal_frames[sig]
            if not frames:
                continue
            combined = pd.concat(frames).sort_index()
            store.put(sig, combined)
            vals = combined.values
            print(f"  /{sig:<10} shape={combined.shape}  "
                  f"min={vals.min():.2f}  mean={vals.mean():.2f}  max={vals.max():.2f}")

    print(f"Done — {os.path.getsize(out_h5)/1e6:.1f} MB")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # January 2026 mode
    ap.add_argument("--jan2026-dir", metavar="DIR",
                    help="Directory containing YYYY-MM-DD subdirs (jan2026 mode)")
    ap.add_argument("--output", default=None,
                    help="Output HDF5 path (default: data/traffic/traffic_jan2026.h5)")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="Keep per-lane columns instead of aggregating to nodes")

    # Legacy mode
    ap.add_argument("--day-type", choices=["weekday", "weekend"], default="weekday")
    ap.add_argument("--extra-runs", nargs="*", metavar="XML", default=[])
    ap.add_argument("--net-file",   default=None)
    ap.add_argument("--no-filter",  action="store_true")

    args = ap.parse_args()

    if args.jan2026_dir:
        input_dir = os.path.join(SIM_DIR, args.jan2026_dir) \
                    if not os.path.isabs(args.jan2026_dir) else args.jan2026_dir
        output_h5 = args.output or os.path.join(SIM_DIR, "data", "traffic",
                                                 "traffic_jan2026.h5")
        net_file  = args.net_file or os.path.join(SIM_DIR, "network", "osm.net.xml")
        aggregate = not args.no_aggregate
        print(f"\nJanuary 2026 mode")
        print(f"  Input dir        : {input_dir}")
        print(f"  Output           : {output_h5}")
        print(f"  Signals          : {', '.join(SIGNALS)}")
        print(f"  Aggregate lanes  : {aggregate}")
        run_jan2026(input_dir, net_file, output_h5, aggregate)
    else:
        run_legacy(args)


if __name__ == "__main__":
    main()
