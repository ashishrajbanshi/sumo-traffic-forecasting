#!/usr/bin/env python3
"""
build_distances.py — Compute directed road-network distances between graph nodes.

Each graph node = one edge midpoint (same definition as build_sensor_locations.py).
Runs forward Dijkstra through the SUMO network topology from every source node
to all reachable nodes within MAX_DISTANCE metres.

Produces:
  data/graph/edges.csv
      from_node, to_node, distance_m
      One row per directed reachable pair.
      Distances are directed: A→B and B→A may differ on one-way roads.

Use edges.csv to build any model's adjacency / weight matrix.

Usage
─────
    python3 script/build_distances.py
    python3 script/build_distances.py --max-distance 1000
"""

import os
import sys
import csv
import heapq
import argparse
import statistics
import xml.etree.ElementTree as ET

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR  = os.path.normpath(os.path.join(BASE_DIR, ".."))

NET_FILE         = os.path.join(SIM_DIR, "network",   "osm.net.xml")
NODE_IDS_FILE    = os.path.join(SIM_DIR, "data",      "graph", "node_ids.txt")
FILTERED_WEEKDAY = os.path.join(SIM_DIR, "detectors", "detectors_filtered_weekday.add.xml")
FILTERED_WEEKEND = os.path.join(SIM_DIR, "detectors", "detectors_filtered_weekend.add.xml")
ALL_WEEKDAY      = os.path.join(SIM_DIR, "detectors", "detectors_weekday.add.xml")
ALL_WEEKEND      = os.path.join(SIM_DIR, "detectors", "detectors_weekend.add.xml")
OUT_CSV          = os.path.join(SIM_DIR, "data",      "graph", "edges.csv")

MAX_DISTANCE = 500.0    # metres — stop Dijkstra beyond this range


# ── Sumolib loader ─────────────────────────────────────────────────────────────

def _load_sumolib():
    for path in [None, "/usr/share/sumo/tools",
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


def lane_to_edge(lane_id):
    return lane_id.rsplit("_", 1)[0]


# ── Node position index ────────────────────────────────────────────────────────

def build_node_positions(det_defs, canonical_nodes):
    """
    Returns {edge_id: pos_m} for each canonical node.
    Uses the first detector found on each edge (all lanes share the same midpoint
    position so any lane's pos_m is representative).
    """
    edge_pos = {}
    for det_id, (lane_id, pos_m) in det_defs.items():
        edge_id = lane_to_edge(lane_id)
        if edge_id in canonical_nodes and edge_id not in edge_pos:
            edge_pos[edge_id] = pos_m
    return edge_pos


# ── Forward Dijkstra ───────────────────────────────────────────────────────────

def dijkstra_from_node(src_edge_id, src_pos_m, node_positions, net, max_dist):
    """
    Walk forward from src_edge_id starting at src_pos_m.
    Returns {tgt_edge_id: road_distance_m} for all reachable canonical nodes.
    src_edge_id itself is excluded from results.
    """
    results  = {}
    src_edge = net.getEdge(src_edge_id)
    dist_to_end = src_edge.getLength() - src_pos_m

    heap    = []
    visited = {src_edge_id}

    for succ in src_edge.getToNode().getOutgoing():
        succ_id = succ.getID()
        if succ_id not in visited and dist_to_end <= max_dist:
            heapq.heappush(heap, (dist_to_end, succ_id))

    while heap:
        dist_at_start, edge_id = heapq.heappop(heap)

        if dist_at_start > max_dist:
            break

        if edge_id in visited:
            continue
        visited.add(edge_id)

        if edge_id in node_positions:
            total_dist = dist_at_start + node_positions[edge_id]
            if total_dist <= max_dist:
                if total_dist < results.get(edge_id, float("inf")):
                    results[edge_id] = total_dist

        edge      = net.getEdge(edge_id)
        dist_next = dist_at_start + edge.getLength()
        if dist_next <= max_dist:
            for succ in edge.getToNode().getOutgoing():
                succ_id = succ.getID()
                if succ_id not in visited:
                    heapq.heappush(heap, (dist_next, succ_id))

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--max-distance", type=float, default=MAX_DISTANCE,
                    help=f"Search radius in metres (default: {MAX_DISTANCE})")
    ap.add_argument("--net-file",  default=NET_FILE)
    ap.add_argument("--node-ids",  default=NODE_IDS_FILE)
    ap.add_argument("--output",    default=OUT_CSV)
    args = ap.parse_args()

    sumolib = _load_sumolib()

    print("\nbuild_distances — directed road-network distances")

    # ── 1. Load canonical node IDs ─────────────────────────────────────────────
    print(f"\n  Reading node IDs : {args.node_ids}")
    with open(args.node_ids, encoding="utf-8") as f:
        canonical_nodes = set(f.read().strip().split(","))
    canonical_nodes.discard("")
    print(f"  Canonical nodes  : {len(canonical_nodes):,}")

    # ── 2. Read detector defs to get midpoint positions per edge ───────────────
    wd_file = _best_add_file(FILTERED_WEEKDAY, ALL_WEEKDAY)
    we_file = _best_add_file(FILTERED_WEEKEND, ALL_WEEKEND)
    print(f"\n  Weekday file : {os.path.basename(wd_file)}")
    wd_defs = read_detector_defs(wd_file)
    print(f"  Weekend file : {os.path.basename(we_file)}")
    we_defs = read_detector_defs(we_file)
    all_defs = {**we_defs, **wd_defs}   # weekday preferred

    node_positions = build_node_positions(all_defs, canonical_nodes)
    missing = canonical_nodes - set(node_positions)
    if missing:
        print(f"  Warning: {len(missing)} nodes have no detector definition (skipped)")
    print(f"  Nodes with positions : {len(node_positions):,}")

    # ── 3. Load SUMO network ───────────────────────────────────────────────────
    print(f"\n  Loading network: {args.net_file}")
    net = sumolib.net.readNet(args.net_file, withInternal=False)

    # ── 4. Forward Dijkstra from each node ────────────────────────────────────
    node_list   = sorted(node_positions.items())
    n           = len(node_list)
    report_step = max(1, n // 20)

    print(f"\n  Dijkstra: {n:,} source nodes | max_dist={args.max_distance:.0f} m")
    print(f"  Progress: ", end="", flush=True)

    rows        = []
    edge_errors = 0

    for i, (src_id, src_pos_m) in enumerate(node_list):
        if i % report_step == 0:
            print(f"{100 * i // n}%…", end="", flush=True)

        try:
            net.getEdge(src_id)
        except Exception:
            edge_errors += 1
            continue

        targets = dijkstra_from_node(src_id, src_pos_m, node_positions,
                                     net, args.max_distance)
        for tgt_id, dist in targets.items():
            rows.append((src_id, tgt_id, round(dist, 2)))

    print("100%")

    if edge_errors:
        print(f"  Warning: {edge_errors} nodes had unresolvable edge IDs and were skipped")

    # ── 5. Write CSV ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"\n  Writing → {args.output}")
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["from_node", "to_node", "distance_m"])
        writer.writerows(rows)

    size_mb = os.path.getsize(args.output) / 1e6

    # ── 6. Summary ─────────────────────────────────────────────────────────────
    dists = [r[2] for r in rows]
    print(f"\nSummary")
    print(f"  Source nodes        : {n:,}")
    print(f"  Directed pairs      : {len(rows):,}")
    avg = len(rows) / n if n else 0
    print(f"  Avg reachable/source: {avg:.1f}")
    if dists:
        print(f"  Distance range      : {min(dists):.1f} – {max(dists):.1f} m")
        print(f"  Distance mean ± std : {statistics.mean(dists):.1f} ± {statistics.stdev(dists):.1f} m")
    print(f"  File size           : {size_mb:.2f} MB")

    print(f"\n  Sample rows:")
    print(f"  {'from_node':<35}  {'to_node':<35}  {'dist_m':>8}")
    for r in rows[:5]:
        print(f"  {r[0]:<35}  {r[1]:<35}  {r[2]:>8.1f}")

    print(f"\nNext steps:")
    print(f"  data/graph/nodes.csv         — node positions (lat/lon)")
    print(f"  data/graph/edges.csv         — directed distances  [this file]")
    print(f"  data/traffic/speed_jan2026.h5 — (time × nodes) speed matrix")
    print(f"  Build model adjacency matrix from edges.csv as needed")


if __name__ == "__main__":
    main()
