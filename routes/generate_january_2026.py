#!/usr/bin/env python3
"""
generate_january_2026.py

Generates randomTrips demand and duarouter routes for every day in
January 2026 (Jan 1 – Jan 30).

  Weekday period : uniform random in [1.60, 1.65] seconds
  Weekend period : uniform random in [1.83, 1.88] seconds

Outputs land in routes/january_2026/:
  trips_2026-01-DD.xml        raw OD trips (randomTrips output)
  routes_2026-01-DD.rou.xml   routed vehicles (duarouter output)

Usage
─────
    cd /mnt/c/Work/simulations/sumo_3/routes
    python3 generate_january_2026.py
"""

import os
import random
import subprocess
import sys
from datetime import date, timedelta

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SIM_DIR      = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

NET_FILE     = os.path.join(SIM_DIR, "network", "osm.net.xml")
OUT_DIR      = os.path.join(SCRIPT_DIR, "january_2026")

SUMO_HOME = os.environ["SUMO_HOME"]
RANDOM_TRIPS = os.path.join(SUMO_HOME, "tools", "randomTrips.py")
DUAROUTER    = "duarouter"

# ── Period ranges ──────────────────────────────────────────────────────────────

WEEKDAY_PERIOD_MIN = 1.60
WEEKDAY_PERIOD_MAX = 1.65

WEEKEND_PERIOD_MIN = 1.83
WEEKEND_PERIOD_MAX = 1.88

# ── Date range ─────────────────────────────────────────────────────────────────

START_DATE = date(2026, 1, 1)
NUM_DAYS   = 30


# ── Helpers ────────────────────────────────────────────────────────────────────

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
    random.seed(42)   # reproducible period choices

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Generating demand + routes for January 2026 ({NUM_DAYS} days)")
    print(f"  Network   : {NET_FILE}")
    print(f"  Output dir: {OUT_DIR}")
    print(f"  Weekday period : [{WEEKDAY_PERIOD_MIN}, {WEEKDAY_PERIOD_MAX}] s")
    print(f"  Weekend period : [{WEEKEND_PERIOD_MIN}, {WEEKEND_PERIOD_MAX}] s")

    for i in range(NUM_DAYS):
        day        = START_DATE + timedelta(days=i)
        date_str   = day.strftime("%Y-%m-%d")
        day_name   = day.strftime("%A")
        is_weekend = day.weekday() >= 5   # Saturday=5, Sunday=6

        if is_weekend:
            period = round(random.uniform(WEEKEND_PERIOD_MIN, WEEKEND_PERIOD_MAX), 4)
        else:
            period = round(random.uniform(WEEKDAY_PERIOD_MIN, WEEKDAY_PERIOD_MAX), 4)

        day_type = "weekend" if is_weekend else "weekday"
        trips_file  = os.path.join(OUT_DIR, f"trips_{date_str}.xml")
        routes_file = os.path.join(OUT_DIR, f"routes_{date_str}.rou.xml")

        print(f"\nDay {i+1:02d}/30  {date_str}  {day_name:<10}  ({day_type})  period={period:.4f}s")

        # ── Step 1: randomTrips ────────────────────────────────────────────────
        run([
            "python3", RANDOM_TRIPS,
            "-n", NET_FILE,
            "-o", trips_file,
            "--prefix", "trip_",
            "--trip-attributes", 'departLane="free" departSpeed="max"',
            "--vehicle-class", "passenger",
            "--fringe-factor", "10",
            "--seed", str(i + 1),
            "--min-distance", "200.0",
            "--end", "86400.0",
            "--period", str(period),
        ], "randomTrips")

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
