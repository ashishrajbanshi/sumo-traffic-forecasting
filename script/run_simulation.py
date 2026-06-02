#!/usr/bin/env python3
"""
run_january_2026.py — Run SUMO mesosim for all 30 days of January 2026.

One simulation per day using its January 2026 route file.
Outputs per day land in:  output/jan2026/YYYY-MM-DD/
  detector_output.xml   — E1 speed/flow measurements (5-min intervals)
  queue_output.xml      — per-edge queue lengths
  tripinfos.xml         — per-vehicle trip statistics
  stats.xml             — simulation-wide summary statistics

Run from sumo_3/:
    python3 script/run_january_2026.py

Options:
    --dry-run     Print commands without executing
    --start DD    Start from day DD (1-30), e.g. --start 5 to resume
    --only DD     Run only day DD
"""

import os
import sys
import shutil
import argparse
import subprocess
from datetime import date, timedelta

# ── Paths (all relative to SIM_DIR) ───────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR  = os.path.normpath(os.path.join(BASE_DIR, ".."))

SUMOCFG_WEEKDAY = os.path.join(SIM_DIR, "config", "sumo_chattanooga_weekdays.sumocfg")
SUMOCFG_WEEKEND = os.path.join(SIM_DIR, "config", "sumo_chattanooga_weekends.sumocfg")

NET_POLY        = os.path.join(SIM_DIR, "network",   "osm.poly.xml.gz")
OUTPUT_ADD      = os.path.join(SIM_DIR, "detectors", "output.add.xml")
DET_WEEKDAY     = os.path.join(SIM_DIR, "detectors", "detectors_weekday.add.xml")
DET_WEEKEND     = os.path.join(SIM_DIR, "detectors", "detectors_weekend.add.xml")
DET_OUT_WEEKDAY = os.path.join(SIM_DIR, "detectors", "detector_output_weekday.xml")
DET_OUT_WEEKEND = os.path.join(SIM_DIR, "detectors", "detector_output_weekend.xml")

ROUTES_DIR = os.path.join(SIM_DIR, "routes", "january_2026")
OUT_BASE   = os.path.join(SIM_DIR, "output", "jan2026")

# ── Date range ─────────────────────────────────────────────────────────────────

START_DATE = date(2026, 1, 1)
NUM_DAYS   = 30


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_day(day, dry_run=False):
    date_str   = day.strftime("%Y-%m-%d")
    is_weekend = day.weekday() >= 5
    day_type   = "weekend" if is_weekend else "weekday"
    day_name   = day.strftime("%A")

    sumocfg    = SUMOCFG_WEEKEND if is_weekend else SUMOCFG_WEEKDAY
    det_add    = DET_WEEKEND     if is_weekend else DET_WEEKDAY
    det_out    = DET_OUT_WEEKEND if is_weekend else DET_OUT_WEEKDAY

    route_file = os.path.join(ROUTES_DIR, f"routes_{date_str}.rou.xml")
    out_dir    = os.path.join(OUT_BASE, date_str)

    # Validate route file exists
    if not os.path.exists(route_file):
        print(f"  ERROR: route file not found: {route_file}")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    # Per-day output paths
    tripinfo_out = os.path.join(out_dir, "tripinfos.xml")
    stats_out    = os.path.join(out_dir, "stats.xml")
    queue_out    = os.path.join(out_dir, "queue_output.xml")
    det_dest     = os.path.join(out_dir, "detector_output.xml")

    cmd = [
        "sumo",
        "-c", sumocfg,
        "--mesosim", "true",
        "--route-files",       route_file,
        "--additional-files",  f"{NET_POLY},{OUTPUT_ADD},{det_add}",
        "--tripinfo-output",   tripinfo_out,
        "--statistic-output",  stats_out,
        "--queue-output",      queue_out,
        "--no-step-log",       "true",
        "--ignore-route-errors", "true",
    ]

    print(f"\n{'─'*70}")
    print(f"  Day {day.timetuple().tm_yday:03d}  {date_str}  {day_name:<10}  ({day_type})")
    print(f"  Route : {os.path.basename(route_file)}")
    print(f"  Output: {out_dir}")

    if dry_run:
        print("  [dry-run] " + " ".join(cmd))
        return True

    result = subprocess.run(cmd, cwd=SIM_DIR, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  SUMO ERROR (exit {result.returncode}):")
        for line in result.stderr.splitlines()[-15:]:
            print(f"    {line}")
        return False

    # Copy detector output to dated directory
    if os.path.exists(det_out):
        shutil.copy2(det_out, det_dest)
        size_mb = os.path.getsize(det_dest) / 1e6
        print(f"  detector_output.xml  : {size_mb:.1f} MB")
    else:
        print(f"  WARNING: detector output not found at {det_out}")

    # Print brief simulation stats
    for line in result.stdout.splitlines():
        if any(k in line for k in ["Inserted", "Loaded", "Running", "ended", "teleport"]):
            print(f"  {line.strip()}")

    q_size = os.path.getsize(queue_out) / 1e6 if os.path.exists(queue_out) else 0
    t_size = os.path.getsize(tripinfo_out) / 1e6 if os.path.exists(tripinfo_out) else 0
    print(f"  queue_output.xml     : {q_size:.1f} MB")
    print(f"  tripinfos.xml        : {t_size:.1f} MB")
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run",  action="store_true",
                    help="Print commands without running SUMO")
    ap.add_argument("--start",    type=int, default=1, metavar="DD",
                    help="Resume from day number (1–30)")
    ap.add_argument("--only",     type=int, default=None, metavar="DD",
                    help="Run only this day number (1–30)")
    args = ap.parse_args()

    print(f"SUMO mesosim — January 2026 ({NUM_DAYS} days)")
    print(f"  Outputs → {OUT_BASE}/YYYY-MM-DD/")

    failed = []
    for i in range(NUM_DAYS):
        day_num = i + 1
        if args.only and day_num != args.only:
            continue
        if day_num < args.start:
            continue

        day = START_DATE + timedelta(days=i)
        ok  = run_day(day, dry_run=args.dry_run)
        if not ok:
            failed.append(day.strftime("%Y-%m-%d"))

    print(f"\n{'═'*70}")
    completed = NUM_DAYS - len(failed)
    if args.only:
        print(f"Done — ran day {args.only}")
    else:
        print(f"Done — {completed}/{NUM_DAYS} days completed")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        print(f"Resume with: python3 script/run_january_2026.py --start <next_day>")


if __name__ == "__main__":
    main()
