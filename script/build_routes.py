#!/usr/bin/env python3
"""
build_routes.py

Generates randomTrips demand and duarouter routes for every day in
January 2026 (Jan 1 – Jan 30).

  Weekday period : uniform random in [1.60, 1.65] seconds
  Weekend period : uniform random in [1.83, 1.88] seconds

Outputs land in routes/january_2026/:
  trips_2026-01-DD.xml        raw OD trips (randomTrips output)
  routes_2026-01-DD.rou.xml   routed vehicles (duarouter output)

Usage
─────
    python3 script/build_routes.py
"""

import os
import subprocess
import sys
from datetime import date, timedelta

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SIM_DIR      = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

NET_FILE     = os.path.join(SIM_DIR, "network", "osm.net.xml")
OUT_DIR      = os.path.join(SIM_DIR, "routes","january_2026")

RANDOM_TRIPS = "/usr/share/sumo/tools/randomTrips.py"
DUAROUTER    = "duarouter"

# ── Time-of-day demand windows ─────────────────────────────────────────────────
# (begin_s, end_s, weekday_period_s, weekend_period_s)
# Lower period = more vehicles = higher demand
TIME_WINDOWS = [
    (0,      6*3600,  12.0,  15.0),   # night       — very sparse
    (6*3600, 9*3600,   0.4,   1.8),   # AM peak     — heavy weekday, light weekend
    (9*3600, 16*3600,  1.8,   1.6),   # midday      — moderate
    (16*3600,19*3600,  0.4,   1.8),   # PM peak     — heavy weekday, light weekend
    (19*3600,24*3600,  2.5,   2.5),   # evening     — light
]

# ── Date range ─────────────────────────────────────────────────────────────────

START_DATE = date(2026, 1, 1)
NUM_DAYS   = 30


# ── Helpers ────────────────────────────────────────────────────────────────────

def _merge_trip_files(src_files, dst_file):
    """Concatenate multiple randomTrips XML outputs into one trips file."""
    import xml.etree.ElementTree as ET
    root = ET.Element("routes")
    for src in src_files:
        tree = ET.parse(src)
        for elem in tree.getroot():
            root.append(elem)
    # Sort by departure time so duarouter gets a clean input
    root[:] = sorted(root, key=lambda e: float(e.get("depart", 0)))
    ET.ElementTree(root).write(dst_file, encoding="unicode", xml_declaration=True)


def run(cmd, label):
    print(f"\n  [{label}]")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR (exit {result.returncode}):")
        for line in result.stderr.splitlines()[-10:]:
            print(f"    {line}")
        sys.exit(1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Generating demand + routes for January 2026 ({NUM_DAYS} days)")
    print(f"  Network   : {NET_FILE}")
    print(f"  Output dir: {OUT_DIR}")
    print(f"  Time windows   : {len(TIME_WINDOWS)} (night/AM-peak/midday/PM-peak/evening)")

    for i in range(NUM_DAYS):
        day        = START_DATE + timedelta(days=i)
        date_str   = day.strftime("%Y-%m-%d")
        day_name   = day.strftime("%A")
        is_weekend = day.weekday() >= 5   # Saturday=5, Sunday=6

        day_type = "weekend" if is_weekend else "weekday"
        trips_file  = os.path.join(OUT_DIR, f"trips_{date_str}.xml")
        routes_file = os.path.join(OUT_DIR, f"routes_{date_str}.rou.xml")

        print(f"\nDay {i+1:02d}/30  {date_str}  {day_name:<10}  ({day_type})")

        # ── Step 1: randomTrips — one run per time window ─────────────────────
        window_files = []
        for w_idx, (begin, end, wd_p, we_p) in enumerate(TIME_WINDOWS):
            period    = wd_p if not is_weekend else we_p
            w_file    = os.path.join(OUT_DIR, f"trips_{date_str}_w{w_idx}.xml")
            prefix    = f"trip_{w_idx}_"
            window_files.append(w_file)
            print(f"  window {w_idx}: {begin//3600:02d}h–{end//3600:02d}h  period={period}s")
            run([
                "python3", RANDOM_TRIPS,
                "-n", NET_FILE,
                "-o", w_file,
                "--prefix", prefix,
                "--trip-attributes", 'departLane="best" departSpeed="avg"',
                "--vehicle-class", "passenger",
                "--fringe-factor", "5",
                "--seed", str(i * 10 + w_idx + 1),
                "--min-distance", "200.0",
                "--begin", str(begin),
                "--end",   str(end),
                "--period", str(period),
            ], f"randomTrips w{w_idx}")

        # Merge all window trip files into one
        _merge_trip_files(window_files, trips_file)
        for f in window_files:
            os.remove(f)

        # ── Step 2: duarouter ─────────────────────────────────────────────────
        run([
            DUAROUTER,
            "-n", NET_FILE,
            "-r", trips_file,
            "-o", routes_file,
            "--ignore-errors", "true",
            "--no-warnings", "true",
        ], "duarouter")

        print(f"  -> {os.path.basename(routes_file)}")

    print(f"\nDone — {NUM_DAYS} days written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
