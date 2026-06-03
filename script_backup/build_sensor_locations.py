#!/usr/bin/env python3
"""
build_sensor_locations.py — Build the graph node table from detector positions.

Each graph NODE = one edge midpoint.  All lane-level detectors on the same edge
share the same midpoint position and collapse into one node:
  - vehicle counts are SUMMED across lanes
  - speed and occupancy are AVERAGED across lanes

Produces:
  data/graph/nodes.csv
      Static columns (always written):
        index, node_id, latitude, longitude,
        lane_count, detector_ids,
        edge_length_m, speed_limit_ms, road_type,
        in_degree, out_degree

      Aggregate columns (populated when --jan2026-dir is given):
        mean_speed_ms       — mean speed across all lanes + all intervals
        total_vehicle_count — total vehicles observed across all lanes + all intervals
        mean_occupancy      — mean occupancy across all lanes + all intervals
        mean_flow_veh_h     — mean flow (veh/hour) across all lanes + all intervals

  data/graph/node_ids.txt
      Comma-separated node IDs (edge IDs) — used by downstream graph builders.

Usage
─────
    python3 script/build_sensor_locations.py
    python3 script/build_sensor_locations.py --union
    python3 script/build_sensor_locations.py --jan2026-dir output/jan2026
"""

import os
import sys
import re
import argparse
import csv
import xml.etree.ElementTree as ET
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR  = os.path.normpath(os.path.join(BASE_DIR, ".."))

NET_FILE         = os.path.join(SIM_DIR, "network",   "osm.net.xml")
FILTERED_WEEKDAY = os.path.join(SIM_DIR, "detectors", "detectors_filtered_weekday.add.xml")
FILTERED_WEEKEND = os.path.join(SIM_DIR, "detectors", "detectors_filtered_weekend.add.xml")
ALL_WEEKDAY      = os.path.join(SIM_DIR, "detectors", "detectors_weekday.add.xml")
ALL_WEEKEND      = os.path.join(SIM_DIR, "detectors", "detectors_weekend.add.xml")
OUT_DIR          = os.path.join(SIM_DIR, "data",      "graph")
OUT_NODES_CSV    = os.path.join(OUT_DIR, "nodes.csv")
OUT_NODE_IDS     = os.path.join(OUT_DIR, "node_ids.txt")

DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


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


def _best_add_file(filtered, fallback):
    if os.path.exists(filtered):
        return filtered
    print(f"  Note: filtered file not found, using unfiltered fallback")
    return fallback


# ── Detector definition reader ─────────────────────────────────────────────────

def read_detector_defs(add_xml_path):
    """Returns {det_id: (lane_id, pos_m)}."""
    defs = {}
    for _, elem in ET.iterparse(add_xml_path, events=("end",)):
        if elem.tag == "e1Detector":
            det_id  = elem.get("id")
            lane_id = elem.get("lane")
            pos     = float(elem.get("pos", 0))
            if det_id and lane_id:
                defs[det_id] = (lane_id, pos)
            elem.clear()
    return defs


# ── Lane → edge ────────────────────────────────────────────────────────────────

def lane_to_edge(lane_id):
    return lane_id.rsplit("_", 1)[0]


# ── Coordinate conversion ──────────────────────────────────────────────────────

_first_lane_error = True

def detector_to_latlon(lane_id, pos_m, net, sumolib):
    global _first_lane_error
    try:
        edge_id  = lane_id.rsplit("_", 1)[0]
        lane_idx = int(lane_id.rsplit("_", 1)[1])
        lane     = net.getEdge(edge_id).getLane(lane_idx)
        shape = lane.getShape()
        if len(shape) < 2:
            x, y = shape[0]
        else:
            x, y = sumolib.geomhelper.positionAtShapeOffset(shape, pos_m)
        lon, lat = net.convertXY2LonLat(x, y)
        return lat, lon
    except Exception as exc:
        if _first_lane_error:
            print(f"  (first lane lookup failure — {lane_id!r}: {exc})")
            _first_lane_error = False
        return None


# ── Aggregate traffic stats from detector output files ─────────────────────────

def aggregate_detector_output(jan2026_dir, det_id_to_edge, canonical_nodes):
    """
    Read all detector_output.xml files under jan2026_dir/YYYY-MM-DD/.

    For each canonical node (edge), aggregate across ALL lanes and ALL intervals:
      - speed     : averaged  (lanes averaged at each interval, then across all intervals)
      - vehicle count : summed (lanes summed at each interval, then across all intervals)
      - occupancy : averaged
      - flow      : averaged

    Handles both meso output (uses 'entered' for count) and
    micro output (uses 'nVehContrib' for count) automatically.

    Returns {edge_id: {mean_speed_ms, total_vehicle_count, mean_occupancy, mean_flow_veh_h}}
    """
    accum = {nid: {
        "speed_sum":   0.0, "speed_count":   0,
        "veh_total":   0,
        "occ_sum":     0.0, "occ_count":     0,
        "flow_sum":    0.0, "flow_count":    0,
    } for nid in canonical_nodes}

    date_dirs = sorted([
        d for d in os.listdir(jan2026_dir)
        if DATE_PATTERN.match(d) and
        os.path.isdir(os.path.join(jan2026_dir, d))
    ])

    if not date_dirs:
        print(f"  Warning: no YYYY-MM-DD directories found in {jan2026_dir}")
        return {nid: {} for nid in canonical_nodes}

    print(f"\n  Aggregating detector output from {len(date_dirs)} days …")

    for date_str in date_dirs:
        day = date_str  # YYYY-MM-DD
        # Simulations write weekday or weekend files; try both.
        for suffix in ("weekday", "weekend"):
            candidate = os.path.join(jan2026_dir, day, f"detector_output_{suffix}.xml")
            if os.path.exists(candidate):
                det_file = candidate
                break
        else:
            print(f"  Warning: missing detector output for {date_str}")
            continue

        for _, elem in ET.iterparse(det_file, events=("end",)):
            if elem.tag != "interval":
                elem.clear()
                continue

            det_id  = elem.get("id")
            edge_id = det_id_to_edge.get(det_id)

            if edge_id not in accum:
                elem.clear()
                continue

            a = accum[edge_id]

            # Speed — valid when >= 0 (both meso and micro use 'speed')
            speed_str = elem.get("speed")
            if speed_str is not None:
                speed = float(speed_str)
                if speed >= 0:
                    a["speed_sum"]   += speed
                    a["speed_count"] += 1

            # Vehicle count — meso uses 'entered'; micro uses 'nVehContrib'
            entered = elem.get("entered")
            if entered is not None:
                a["veh_total"] += int(float(entered))
            else:
                nveh = elem.get("nVehContrib")
                if nveh is not None:
                    a["veh_total"] += int(float(nveh))

            # Occupancy
            occ_str = elem.get("occupancy")
            if occ_str is not None:
                occ = float(occ_str)
                if occ >= 0:
                    a["occ_sum"]   += occ
                    a["occ_count"] += 1

            # Flow (vehicles per hour)
            flow_str = elem.get("flow")
            if flow_str is not None:
                flow = float(flow_str)
                if flow >= 0:
                    a["flow_sum"]   += flow
                    a["flow_count"] += 1

            elem.clear()

    result = {}
    for edge_id, a in accum.items():
        result[edge_id] = {
            "mean_speed_ms":       round(a["speed_sum"] / a["speed_count"], 4)
                                   if a["speed_count"] > 0 else None,
            "total_vehicle_count": a["veh_total"],
            "mean_occupancy":      round(a["occ_sum"] / a["occ_count"], 4)
                                   if a["occ_count"] > 0 else None,
            "mean_flow_veh_h":     round(a["flow_sum"] / a["flow_count"], 4)
                                   if a["flow_count"] > 0 else None,
        }
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--union", action="store_true",
                    help="Use union of weekday+weekend instead of intersection")
    ap.add_argument("--net-file",    default=NET_FILE)
    ap.add_argument("--out-nodes",   default=OUT_NODES_CSV)
    ap.add_argument("--out-ids",     default=OUT_NODE_IDS)
    ap.add_argument("--jan2026-dir", default=None,
                    help="Path to output/jan2026 to compute per-node aggregate stats")
    args = ap.parse_args()

    sumolib = _load_sumolib()

    print("\nbuild_sensor_locations — graph node table")

    # ── 1. Read detector definitions ───────────────────────────────────────────
    wd_file = _best_add_file(FILTERED_WEEKDAY, ALL_WEEKDAY)
    we_file = _best_add_file(FILTERED_WEEKEND, ALL_WEEKEND)

    print(f"\n  Weekday file : {os.path.basename(wd_file)}")
    wd_defs = read_detector_defs(wd_file)
    print(f"  Weekend file : {os.path.basename(we_file)}")
    we_defs = read_detector_defs(we_file)

    print(f"\n  Weekday detectors : {len(wd_defs):,}")
    print(f"  Weekend detectors : {len(we_defs):,}")

    # ── 2. Build canonical detector set ────────────────────────────────────────
    if args.union:
        canonical_det_ids = set(wd_defs) | set(we_defs)
        set_label = "union"
    else:
        canonical_det_ids = set(wd_defs) & set(we_defs)
        set_label = "intersection"

    all_defs = {**we_defs, **wd_defs}
    print(f"  Canonical detectors ({set_label}): {len(canonical_det_ids):,}")

    # ── 3. Group detectors by edge → nodes ────────────────────────────────────
    edge_to_dets   = defaultdict(list)
    det_id_to_edge = {}
    for det_id in canonical_det_ids:
        lane_id, pos_m = all_defs[det_id]
        edge_id = lane_to_edge(lane_id)
        edge_to_dets[edge_id].append((det_id, lane_id, pos_m))
        det_id_to_edge[det_id] = edge_id

    node_ids = sorted(edge_to_dets.keys())
    print(f"  Graph nodes (unique edges): {len(node_ids):,}")

    # ── 4. Load network ────────────────────────────────────────────────────────
    print(f"\n  Loading network: {args.net_file}")
    net = sumolib.net.readNet(args.net_file, withInternal=False)

    # ── 5. Compute static properties per node ─────────────────────────────────
    print(f"\n  Computing node properties for {len(node_ids):,} nodes …")

    rows    = []
    skipped = []
    lat_min = lat_max = lon_min = lon_max = None

    for node_id in node_ids:
        dets    = edge_to_dets[node_id]
        _, lane_id, pos_m = dets[0]   # all lanes share the same midpoint position

        # GPS position
        result = detector_to_latlon(lane_id, pos_m, net, sumolib)
        if result is None:
            skipped.append(node_id)
            continue
        lat, lon = result

        # Static network properties
        try:
            edge          = net.getEdge(node_id)
            edge_length_m = round(edge.getLength(), 2)
            speed_limit   = round(edge.getSpeed(), 4)   # m/s
            road_type     = edge.getType() or ""
            in_degree     = len(edge.getFromNode().getIncoming())
            out_degree    = len(edge.getToNode().getOutgoing())
        except Exception:
            edge_length_m = None
            speed_limit   = None
            road_type     = ""
            in_degree     = None
            out_degree    = None

        lane_count   = len(dets)
        detector_ids = "|".join(sorted(d[0] for d in dets))

        rows.append({
            "node_id":             node_id,
            "latitude":            lat,
            "longitude":           lon,
            "lane_count":          lane_count,
            "detector_ids":        detector_ids,
            "edge_length_m":       edge_length_m,
            "speed_limit_ms":      speed_limit,
            "road_type":           road_type,
            "in_degree":           in_degree,
            "out_degree":          out_degree,
            # aggregate stats — filled below if --jan2026-dir provided
            "mean_speed_ms":       None,
            "total_vehicle_count": None,
            "mean_occupancy":      None,
            "mean_flow_veh_h":     None,
        })

        if lat_min is None:
            lat_min = lat_max = lat
            lon_min = lon_max = lon
        else:
            lat_min = min(lat_min, lat)
            lat_max = max(lat_max, lat)
            lon_min = min(lon_min, lon)
            lon_max = max(lon_max, lon)

    if skipped:
        print(f"  Warning: {len(skipped)} nodes skipped (lane not found in network)")

    # ── 6. Aggregate traffic stats from detector output files (optional) ────────
    if args.jan2026_dir:
        jan_dir = os.path.join(SIM_DIR, args.jan2026_dir) \
                  if not os.path.isabs(args.jan2026_dir) else args.jan2026_dir
        canonical_nodes = {r["node_id"] for r in rows}
        agg = aggregate_detector_output(jan_dir, det_id_to_edge, canonical_nodes)
        for r in rows:
            a = agg.get(r["node_id"], {})
            r["mean_speed_ms"]       = a.get("mean_speed_ms")
            r["total_vehicle_count"] = a.get("total_vehicle_count")
            r["mean_occupancy"]      = a.get("mean_occupancy")
            r["mean_flow_veh_h"]     = a.get("mean_flow_veh_h")
        filled = sum(1 for r in rows if r["mean_speed_ms"] is not None)
        print(f"  Aggregate stats attached: {filled:,} / {len(rows):,} nodes")
    else:
        print(f"\n  Note: run with --jan2026-dir output/jan2026 to populate aggregate columns.")

    # ── 7. Write outputs ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out_nodes), exist_ok=True)

    columns = [
        "index", "node_id",
        "latitude", "longitude",
        "lane_count", "detector_ids",
        "edge_length_m", "speed_limit_ms", "road_type",
        "in_degree", "out_degree",
        "mean_speed_ms", "total_vehicle_count", "mean_occupancy", "mean_flow_veh_h",
    ]

    with open(args.out_nodes, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i, r in enumerate(rows):
            r["index"] = i
            writer.writerow(r)

    print(f"\n  Written → {args.out_nodes}")

    with open(args.out_ids, "w", encoding="utf-8") as f:
        f.write(",".join(r["node_id"] for r in rows))
    print(f"  Written → {args.out_ids}")

    # ── 8. Summary ─────────────────────────────────────────────────────────────
    lane_counts = [r["lane_count"] for r in rows]
    multi_lane  = sum(1 for c in lane_counts if c > 1)

    print(f"\nSummary")
    print(f"  Nodes written    : {len(rows):,}")
    print(f"  Nodes skipped    : {len(skipped)}")
    if rows:
        print(f"  Multi-lane nodes : {multi_lane:,} "
              f"({100*multi_lane/len(rows):.1f}% — lanes collapsed to 1 node)")
        print(f"  Single-lane nodes: {len(rows) - multi_lane:,}")
    if lat_min is not None:
        print(f"  Bounding box     : lat [{lat_min:.5f}, {lat_max:.5f}]")
        print(f"                     lon [{lon_min:.5f}, {lon_max:.5f}]")

    guard_lat = (34.980, 35.100)
    guard_lon = (-85.440, -85.240)
    oob = [r for r in rows
           if not (guard_lat[0] <= r["latitude"] <= guard_lat[1] and
                   guard_lon[0] <= r["longitude"] <= guard_lon[1])]
    if oob:
        print(f"\n  WARNING: {len(oob)} nodes outside Chattanooga metro guard bbox")
    else:
        print(f"  Bbox check       : all {len(rows):,} nodes inside Chattanooga metro area ✓")

    print(f"\n  Sample nodes:")
    print(f"  {'idx':>4}  {'node_id':<28}  {'lat':>10}  {'lon':>11}  "
          f"{'lanes':>5}  {'len_m':>7}  {'spd_lim':>7}  {'type':<25}")
    for i, r in enumerate(rows[:5]):
        print(f"  {i:>4}  {r['node_id']:<28}  {r['latitude']:>10.6f}  {r['longitude']:>11.6f}  "
              f"{r['lane_count']:>5}  {str(r['edge_length_m']):>7}  "
              f"{str(r['speed_limit_ms']):>7}  {r['road_type']:<25}")


if __name__ == "__main__":
    main()
