#!/usr/bin/env python3
"""
build_routes.py

Generates time-varying demand routes for every day in January 2026.
Uses five time windows per day to simulate rush-hour peaks.

Outputs land in routes/january_2026/:
  routes_2026-01-DD.rou.xml   routed vehicles

Usage
─────
    python3 script/build_routes.py
"""

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import date, timedelta

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR    = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
NET_FILE   = os.path.join(SIM_DIR, "network", "osm.net.xml")
OUT_DIR    = os.path.join(SIM_DIR, "routes", "january_2026")


def _find_random_trips():
    candidates = [
        os.path.join(os.environ.get("SUMO_HOME", ""), "tools", "randomTrips.py"),
        "/usr/share/sumo/tools/randomTrips.py",
        "/usr/local/share/sumo/tools/randomTrips.py",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    sys.exit("ERROR: randomTrips.py not found. Set SUMO_HOME or install SUMO.")


RANDOM_TRIPS = _find_random_trips()

# ── Time-of-day demand windows ─────────────────────────────────────────────────
# (begin_s, end_s, weekday_period_s, weekend_period_s)
# Lower period = more vehicles spawned per second = heavier demand

TIME_WINDOWS = [
    (0,        6*3600,  12.0,  15.0),   # night        — very sparse
    (6*3600,   9*3600,   0.4,   1.8),   # AM peak      — heavy weekday
    (9*3600,  16*3600,   1.8,   1.6),   # midday       — moderate
    (16*3600, 19*3600,   0.4,   1.8),   # PM peak      — heavy weekday
    (19*3600, 24*3600,   2.5,   2.5),   # evening      — light
]

START_DATE = date(2026, 1, 1)
NUM_DAYS   = 30


# ── Helpers ────────────────────────────────────────────────────────────────────

def run(cmd, label):
    print(f"  [{label}] " + " ".join(cmd[-6:]))   # print last few args only
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR (exit {result.returncode}):")
        for line in result.stderr.splitlines()[-10:]:
            print(f"    {line}")
        sys.exit(1)


def merge_route_files(src_files, dst_file):
    """Merge per-window route files into one, sorted by depart time."""
    root = ET.Element("routes")
    for src in src_files:
        if not os.path.exists(src):
            continue
        for elem in ET.parse(src).getroot():
            if elem.tag in ("vehicle", "trip", "flow"):
                root.append(elem)
    root[:] = sorted(root, key=lambda e: float(e.get("depart", 0)))
    ET.ElementTree(root).write(dst_file, encoding="unicode", xml_declaration=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Generating demand + routes for January 2026 ({NUM_DAYS} days)")
    print(f"  Network    : {NET_FILE}")
    print(f"  Output dir : {OUT_DIR}")
    print(f"  randomTrips: {RANDOM_TRIPS}")
    print(f"  Time windows: {len(TIME_WINDOWS)} (night/AM-peak/midday/PM-peak/evening)")

    for i in range(NUM_DAYS):
        day        = START_DATE + timedelta(days=i)
        date_str   = day.strftime("%Y-%m-%d")
        day_name   = day.strftime("%A")
        is_weekend = day.weekday() >= 5

        day_type    = "weekend" if is_weekend else "weekday"
        routes_file = os.path.join(OUT_DIR, f"routes_{date_str}.rou.xml")

        print(f"\nDay {i+1:02d}/30  {date_str}  {day_name:<10}  ({day_type})")

        window_route_files = []
        window_trip_files  = []

        for w_idx, (begin, end, wd_p, we_p) in enumerate(TIME_WINDOWS):
            period   = wd_p if not is_weekend else we_p
            w_trips  = os.path.join(OUT_DIR, f"_tmp_{date_str}_w{w_idx}.trips.xml")
            w_routes = os.path.join(OUT_DIR, f"_tmp_{date_str}_w{w_idx}.rou.xml")
            window_trip_files.append(w_trips)
            window_route_files.append(w_routes)

            print(f"  window {w_idx}: {begin//3600:02d}h–{end//3600:02d}h  period={period}s")

            # SUMO 1.26 randomTrips reads the route file before writing it
            with open(w_routes, "w") as f:
                f.write('<routes/>\n')

            run([
                "python3", RANDOM_TRIPS,
                "-n",              NET_FILE,
                "-o",              w_trips,
                "--route-file",    w_routes,
                "--prefix",        f"t{w_idx}_",
                "--trip-attributes", 'departLane="best" departSpeed="avg"',
                "--vehicle-class", "passenger",
                "--fringe-factor", "5",
                "--seed",          str(i * 10 + w_idx + 1),
                "--min-distance",  "200.0",
                "--begin",         str(begin),
                "--end",           str(end),
                "--period",        str(period),
            ], f"randomTrips w{w_idx}")

        # Merge all window route files into one final route file
        merge_route_files(window_route_files, routes_file)

        # Clean up temp files
        for f in window_trip_files + window_route_files:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass

        n_vehicles = sum(1 for _ in ET.parse(routes_file).getroot())
        print(f"  -> {os.path.basename(routes_file)}  ({n_vehicles:,} vehicles)")

    print(f"\nDone — {NUM_DAYS} days written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
