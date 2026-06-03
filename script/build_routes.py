#!/usr/bin/env python3
"""
build_routes.py

Generates time-varying demand routes for every day in January 2026.
Bypasses randomTrips.py entirely (avoids SUMO version compatibility issues)
by generating trip XML directly via sumolib, then routing with duarouter.

Five time windows per day simulate rush-hour peaks:
  night / AM-peak / midday / PM-peak / evening

Usage
─────
    python3 script/build_routes.py
"""

import os
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import date, timedelta

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR    = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
NET_FILE   = os.path.join(SIM_DIR, "network", "osm.net.xml")
OUT_DIR    = os.path.join(SIM_DIR, "routes", "january_2026")
DUAROUTER  = "duarouter"

# ── Time-of-day demand windows ─────────────────────────────────────────────────
# (begin_s, end_s, weekday_period_s, weekend_period_s)
# period = seconds between vehicle spawns — lower = heavier demand

TIME_WINDOWS = [
    (0,        6*3600,  12.0,  15.0),   # night
    (6*3600,   9*3600,   0.4,   1.8),   # AM peak
    (9*3600,  16*3600,   1.8,   1.6),   # midday
    (16*3600, 19*3600,   0.4,   1.8),   # PM peak
    (19*3600, 24*3600,   2.5,   2.5),   # evening
]

MIN_EDGE_LENGTH = 100.0   # metres — edges shorter than this are skipped
START_DATE      = date(2026, 1, 1)
NUM_DAYS        = 30


# ── Network edge loader ────────────────────────────────────────────────────────

def load_valid_edges(net_file):
    """
    Return list of edge IDs suitable as trip endpoints.
    Uses sumolib; filters to passenger-allowed, non-internal edges >= MIN_EDGE_LENGTH.
    """
    sumo_home = os.environ.get("SUMO_HOME", "")
    for path in [
        os.path.join(sumo_home, "tools"),
        "/usr/share/sumo/tools",
        "/usr/local/share/sumo/tools",
    ]:
        if path and os.path.isdir(path):
            sys.path.insert(0, path)

    try:
        import sumolib
    except ImportError:
        sys.exit("ERROR: sumolib not found. Activate sumo_env or set SUMO_HOME.")

    print(f"  Loading network …")
    net   = sumolib.net.readNet(net_file, withInternal=False)
    edges = [
        e.getID() for e in net.getEdges()
        if e.allows("passenger")
        and e.getLength() >= MIN_EDGE_LENGTH
        and not e.getID().startswith(":")
    ]
    print(f"  Valid edges for demand: {len(edges):,}")
    return edges


# ── Trip XML writer ────────────────────────────────────────────────────────────

def write_trips_xml(edges, time_windows, is_weekend, seed, out_file):
    """
    Generate trip XML with Poisson-sampled departure times per time window.
    Writes directly — no randomTrips.py needed.
    """
    rng   = random.Random(seed)
    trips = []   # list of (depart_s, trip_id, from_edge, to_edge)

    for w_idx, (begin, end, wd_p, we_p) in enumerate(time_windows):
        period     = wd_p if not is_weekend else we_p
        n_vehicles = max(1, int((end - begin) / period))

        for v in range(n_vehicles):
            depart   = rng.uniform(begin, end - 1)
            from_e   = rng.choice(edges)
            to_e     = rng.choice(edges)
            while to_e == from_e:
                to_e = rng.choice(edges)
            trips.append((depart, f"t{w_idx}_{v}", from_e, to_e))

    trips.sort(key=lambda t: t[0])

    with open(out_file, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes>\n')
        f.write('    <vType id="passenger" vClass="passenger"/>\n')
        for depart, vid, frm, to in trips:
            f.write(
                f'    <trip id="{vid}" type="passenger" depart="{depart:.2f}" '
                f'from="{frm}" to="{to}" '
                f'departLane="best" departSpeed="avg"/>\n'
            )
        f.write('</routes>\n')

    return len(trips)


# ── Helpers ────────────────────────────────────────────────────────────────────

def run_duarouter(trips_file, routes_file, net_file):
    cmd = [
        DUAROUTER,
        "-n", net_file,
        "-r", trips_file,
        "-o", routes_file,
        "--ignore-errors", "true",
        "--no-warnings",   "true",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR duarouter (exit {result.returncode}):")
        for line in result.stderr.splitlines()[-10:]:
            print(f"    {line}")
        sys.exit(1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Generating demand + routes for January 2026 ({NUM_DAYS} days)")
    print(f"  Network    : {NET_FILE}")
    print(f"  Output dir : {OUT_DIR}")
    print(f"  Time windows: {len(TIME_WINDOWS)} (night/AM-peak/midday/PM-peak/evening)")

    edges = load_valid_edges(NET_FILE)

    for i in range(NUM_DAYS):
        day        = START_DATE + timedelta(days=i)
        date_str   = day.strftime("%Y-%m-%d")
        day_name   = day.strftime("%A")
        is_weekend = day.weekday() >= 5
        day_type   = "weekend" if is_weekend else "weekday"

        trips_file  = os.path.join(OUT_DIR, f"trips_{date_str}.xml")
        routes_file = os.path.join(OUT_DIR, f"routes_{date_str}.rou.xml")

        print(f"\nDay {i+1:02d}/30  {date_str}  {day_name:<10}  ({day_type})")

        n = write_trips_xml(edges, TIME_WINDOWS, is_weekend,
                            seed=i + 1, out_file=trips_file)
        print(f"  Generated {n:,} trips → {os.path.basename(trips_file)}")

        run_duarouter(trips_file, routes_file, NET_FILE)
        os.remove(trips_file)   # clean up raw trips

        n_routed = sum(1 for _ in ET.parse(routes_file).getroot()
                       if _.tag == "vehicle")
        print(f"  Routed    {n_routed:,} vehicles → {os.path.basename(routes_file)}")

    print(f"\nDone — {NUM_DAYS} days written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
