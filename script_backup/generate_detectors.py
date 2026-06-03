#!/usr/bin/env python3
"""
generate_detectors.py — E1 induction-loop detector generator for the
Chattanooga SUMO network.

Placement strategy
──────────────────
One detector per lane, placed at the exact midpoint of the lane
(pos = lane_length / 2).

Because all lanes on the same edge share the same length, every lane
on a given edge gets a detector at the same position.  A 3-lane section
of 300 m produces three detectors all at pos=150 m — spatially aligned
and representing the same road cross-section.  This alignment allows
downstream GNN stages to optionally collapse multi-lane sensors at the
same location into a single node (by averaging or aggregating speeds).

All lanes of each qualifying edge receive exactly one detector.

Workflow
────────
  1.  python3 script/generate_detectors.py    → detectors_weekday/weekend.add.xml
  2.  Run SUMO once (a short 3 600-s test is enough)
  3.  python3 script/filter_detectors.py      → detectors_filtered_*.add.xml
  4.  For long training runs, swap to the filtered file in the .sumocfg.
"""

import os
import sys
import math  # used for _shape_len

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR  = os.path.normpath(os.path.join(BASE_DIR, ".."))

NET_FILE = os.path.join(SIM_DIR, "network", "osm.net.xml")

# Two output files — one per day type; file attr is relative to detectors/ dir.
OUTPUTS = {
    "weekday": (
        os.path.join(SIM_DIR, "detectors", "detectors_weekday.add.xml"),
        "detector_output_weekday.xml",
    ),
    "weekend": (
        os.path.join(SIM_DIR, "detectors", "detectors_weekend.add.xml"),
        "detector_output_weekend.xml",
    ),
}

# ── Parameters ────────────────────────────────────────────────────────────────

# Data collection interval: 300 s = 5 min (matches METR-LA / PeMS standard)
FREQ = 300

# Edges shorter than this are too short for meaningful measurement.
MIN_LANE_LEN_M = 20.0

# Qualifying road types — same set as before; only these classes get detectors.
ROAD_TYPES = {
    "highway.motorway",
    "highway.motorway_link",
    "highway.trunk",
    "highway.trunk_link",
    "highway.primary",
    "highway.primary_link",
    "highway.secondary",
    "highway.secondary_link",
}


# ── sumolib loader ────────────────────────────────────────────────────────────

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
    print("ERROR: sumolib not found.\n"
          "  pip install sumolib  — or —  set SUMO_HOME")
    sys.exit(1)


# ── Geometry ──────────────────────────────────────────────────────────────────

def _shape_len(shape):
    total = 0.0
    for i in range(len(shape) - 1):
        dx = shape[i+1][0] - shape[i][0]
        dy = shape[i+1][1] - shape[i][1]
        total += math.sqrt(dx*dx + dy*dy)
    return total


def _effective_len(lane):
    """min(SUMO stored length, shape-derived length) — never overshoot the lane."""
    sumo_len = lane.getLength()
    shape    = lane.getShape()
    geom_len = _shape_len(shape) if len(shape) >= 2 else sumo_len
    return min(sumo_len, geom_len)


# ── Road-type matching ────────────────────────────────────────────────────────

def _matched_type(edge):
    """Return True if this edge's type is in ROAD_TYPES, else False."""
    for part in (edge.getType() or "").split("|"):
        if part.strip() in ROAD_TYPES:
            return True
    return False


# ── Midpoint placement ────────────────────────────────────────────────────────

def _midpoint(lane_len):
    """Single detector at the lane midpoint."""
    return round(lane_len / 2.0, 2)


# ── XML ID sanitisation ───────────────────────────────────────────────────────

def _xml_id(lane_id, pos_m):
    """
    Valid XML NCName from a SUMO lane ID + position.
    '-' → 'n',  '#' → 's',  '.' → 'd',  ':' → 'c'
    """
    clean = (lane_id
             .replace("-", "n")
             .replace("#", "s")
             .replace(".", "d")
             .replace(":", "c"))
    return f"det_{clean}_{int(pos_m):05d}"


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(net_file=NET_FILE):
    sumolib = _load_sumolib()

    if not os.path.exists(net_file):
        print(f"ERROR: Network file not found:\n  {net_file}")
        sys.exit(1)

    print(f"Reading network …\n  {net_file}")
    net = sumolib.net.readNet(net_file, withInternal=False)

    detectors  = []       # (id, lane_id, pos)
    type_stats = {}       # road_type → (edges, lanes, detectors)
    skip_type  = 0
    skip_short = 0

    for edge in net.getEdges():
        if not _matched_type(edge):
            skip_type += 1
            continue

        lanes = edge.getLanes()
        if not lanes:
            continue

        # All lanes on the same edge share the same length, so midpoint is
        # identical across all lanes — they are spatially aligned.
        edge_type  = next(
            (p.strip() for p in (edge.getType() or "").split("|") if p.strip() in ROAD_TYPES),
            "unknown"
        )
        e_count, l_count, d_count = type_stats.get(edge_type, (0, 0, 0))
        edge_had_lane = False

        for lane in lanes:
            lane_len = _effective_len(lane)
            if lane_len < MIN_LANE_LEN_M:
                skip_short += 1
                continue

            pos     = _midpoint(lane_len)
            lane_id = lane.getID()
            detectors.append((_xml_id(lane_id, pos), lane_id, pos))

            l_count      += 1
            d_count      += 1
            edge_had_lane = True

        if edge_had_lane:
            e_count += 1

        type_stats[edge_type] = (e_count, l_count, d_count)

    # ── Report ──────────────────────────────────────────────────────────────
    print()
    print(f"  {'Road type':<35s} {'Edges':>6s}  {'Lanes':>6s}  {'Detectors':>10s}")
    print(f"  {'-'*63}")
    total_e = total_l = total_d = 0
    for rt in sorted(type_stats):
        e, l, d = type_stats[rt]
        print(f"  {rt:<35s} {e:>6d}  {l:>6d}  {d:>10d}")
        total_e += e;  total_l += l;  total_d += d
    print(f"  {'-'*63}")
    print(f"  {'TOTAL':<35s} {total_e:>6d}  {total_l:>6d}  {total_d:>10d}")
    print()
    print(f"  Edges skipped (non-target road type)  : {skip_type}")
    print(f"  Lanes skipped (< {MIN_LANE_LEN_M:.0f} m)             : {skip_short}")
    print()

    for day_type, (out_file, det_data_file) in OUTPUTS.items():
        _write_xml(detectors, out_file, total_d, det_data_file)
        print(f"  [{day_type}] → {out_file}")
    return len(detectors)


def _write_xml(detectors, path, count, det_data_file):
    header_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!-- E1 induction-loop detectors — Chattanooga SUMO network",
        "     Generator  : script/generate_detectors.py",
        "     Method     : midpoint placement — one detector per lane at lane_len/2",
        "     Policy     : all lanes on qualifying edges; sensors on parallel lanes",
        "                  of the same edge share the same position (spatially aligned)",
        "     Road types : " + ", ".join(sorted(ROAD_TYPES)),
        f"     Min length : {MIN_LANE_LEN_M} m",
        f"     Interval   : {FREQ} s (5 min)",
        f"     Detectors  : {count}",
        f"     Output     : detectors/{det_data_file}",
        "     Next step  : python3 script/filter_detectors.py  (after a test run)",
        "-->",
        '<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '            xsi:noNamespaceSchemaLocation='
            '"http://sumo.dlr.de/xsd/additional_file.xsd">',
    ]
    entry_lines = [
        f'    <e1Detector id="{did}" lane="{lid}" pos="{pos}"'
        f' freq="{FREQ}" file="{det_data_file}"/>'
        for did, lid, pos in detectors
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines + entry_lines + ["</additional>\n"]))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n = generate()
    print(f"Done — {n} detectors written to weekday + weekend definition files.")
    print()
    print("Next steps:")
    print("  1. Short test run — weekday:")
    print("       sumo -c config/sumo_chattanooga_weekdays.sumocfg --end 3600")
    print("  2. Filter weekday detectors:")
    print("       python3 script/filter_detectors.py --day-type weekday")
    print("  3. Short test run — weekend:")
    print("       sumo -c config/sumo_chattanooga.sumocfg --end 3600")
    print("  4. Filter weekend detectors:")
    print("       python3 script/filter_detectors.py --day-type weekend")
