#!/usr/bin/env python3
"""
run_simulation.py — Run SUMO mesosim for all 30 days of January 2026 in parallel.

Each day runs as a separate SUMO subprocess. With --workers N, up to N days run
simultaneously. On a 32-core HPC all 30 days can run at once (~30x speedup).

Outputs per day land in:  output/jan2026/YYYY-MM-DD/
  detector_output_weekday.xml  — E1 speed/flow measurements (5-min intervals)
  tripinfos.xml                — per-vehicle trip statistics
  stats.xml                    — simulation-wide summary statistics
  queue_output.xml             — per-edge queue lengths

Run from sumo_3/:
    python3 script/run_simulation.py                  # all 30 days, all cores
    python3 script/run_simulation.py --workers 16     # limit to 16 parallel
    python3 script/run_simulation.py --start 5        # resume from day 5
    python3 script/run_simulation.py --only 12        # run only day 12
    python3 script/run_simulation.py --dry-run        # print commands, no exec
"""

import argparse
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR  = os.path.normpath(os.path.join(BASE_DIR, ".."))

SUMOCFG_WEEKDAY = os.path.join(SIM_DIR, "config", "sumo_chattanooga_weekdays.sumocfg")
SUMOCFG_WEEKEND = os.path.join(SIM_DIR, "config", "sumo_chattanooga_weekends.sumocfg")
DET_WEEKDAY     = os.path.join(SIM_DIR, "detectors", "detectors_weekday.add.xml")
DET_WEEKEND     = os.path.join(SIM_DIR, "detectors", "detectors_weekend.add.xml")

ROUTES_DIR = os.path.join(SIM_DIR, "routes", "january_2026")
OUT_BASE   = os.path.join(SIM_DIR, "output",  "jan2026")

START_DATE = date(2026, 1, 1)
NUM_DAYS   = 30


# ── Single-day runner ──────────────────────────────────────────────────────────

def _make_temp_det_file(det_add: str, day_type: str, abs_output: str) -> str:
    """Return path to a temp additional file with the detector output path set to abs_output."""
    with open(det_add) as f:
        content = f.read()
    content = content.replace(
        f'file="detector_output_{day_type}.xml"',
        f'file="{abs_output}"',
    )
    fd, tmp_path = tempfile.mkstemp(suffix=".add.xml", dir=os.path.dirname(det_add))
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return tmp_path


def run_day(day: date, dry_run: bool = False) -> tuple:
    """Run SUMO for one day. Returns (date_str, success, message)."""
    date_str   = day.strftime("%Y-%m-%d")
    is_weekend = day.weekday() >= 5
    day_type   = "weekend" if is_weekend else "weekday"
    sumocfg    = SUMOCFG_WEEKEND if is_weekend else SUMOCFG_WEEKDAY
    det_add    = DET_WEEKEND     if is_weekend else DET_WEEKDAY

    route_file = os.path.join(ROUTES_DIR, f"routes_{date_str}.rou.xml")
    out_dir    = os.path.join(OUT_BASE, date_str)

    if not os.path.exists(route_file):
        return date_str, False, f"route file not found: {route_file}"

    os.makedirs(out_dir, exist_ok=True)

    # Write a per-day temp additional file with the absolute detector output path.
    # --output-prefix can't be used here: SUMO string-concatenates the additional
    # file's directory with the prefix, producing a doubled path when the prefix
    # is absolute (e.g. detectors//abs/path/file.xml).
    det_out = os.path.join(out_dir, f"detector_output_{day_type}.xml")
    tmp_det_add = _make_temp_det_file(det_add, day_type, det_out)

    cmd = [
        "sumo",
        "-c",                    sumocfg,
        "--mesosim",             "true",
        "--route-files",         route_file,
        "--additional-files",    tmp_det_add,
        "--tripinfo-output",     os.path.join(out_dir, "tripinfos.xml"),
        "--statistic-output",    os.path.join(out_dir, "stats.xml"),
        "--queue-output",        os.path.join(out_dir, "queue_output.xml"),
        "--no-step-log",         "true",
        "--ignore-route-errors", "true",
    ]

    if dry_run:
        os.unlink(tmp_det_add)
        return date_str, True, "  [dry-run] " + " ".join(cmd)

    try:
        result = subprocess.run(cmd, cwd=SIM_DIR, capture_output=True, text=True)
    finally:
        try:
            os.unlink(tmp_det_add)
        except OSError:
            pass

    if result.returncode != 0:
        last_lines = "\n".join(result.stderr.splitlines()[-10:])
        return date_str, False, f"SUMO error (exit {result.returncode}):\n{last_lines}"

    summary_lines = [
        line.strip() for line in result.stdout.splitlines()
        if any(k in line for k in ["Inserted", "Loaded", "ended", "teleport"])
    ]

    det_file = os.path.join(out_dir, f"detector_output_{day_type}.xml")
    if os.path.exists(det_file):
        size_mb = os.path.getsize(det_file) / 1e6
        summary_lines.append(f"detector_output_{day_type}.xml: {size_mb:.1f} MB")
    else:
        summary_lines.append("WARNING: detector output not found")

    return date_str, True, "\n  ".join(summary_lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--workers", type=int, default=None,
                    help="Max parallel SUMO processes (default: number of CPU cores)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without running SUMO")
    ap.add_argument("--start",   type=int, default=1, metavar="DD",
                    help="Resume from day number (1–30)")
    ap.add_argument("--only",    type=int, default=None, metavar="DD",
                    help="Run only this day number (1–30)")
    args = ap.parse_args()

    days = []
    for i in range(NUM_DAYS):
        day_num = i + 1
        if args.only and day_num != args.only:
            continue
        if day_num < args.start:
            continue
        days.append(START_DATE + timedelta(days=i))

    n_workers = args.workers or min(len(days), os.cpu_count() or 1)

    print(f"SUMO mesosim — January 2026")
    print(f"  Days      : {len(days)}")
    print(f"  Workers   : {n_workers}  (cores available: {os.cpu_count()})")
    print(f"  Outputs   : {OUT_BASE}/YYYY-MM-DD/")

    # Sequential fallback for single day or dry-run
    if len(days) == 1 or args.dry_run:
        failed = []
        for day in days:
            date_str, ok, msg = run_day(day, dry_run=args.dry_run)
            status = "OK " if ok else "ERR"
            print(f"\n[{status}] {date_str}")
            print(f"  {msg}")
            if not ok:
                failed.append(date_str)
        _print_summary(days, failed)
        return

    # Parallel execution
    print(f"\nStarting parallel simulation ...\n")
    failed   = []
    finished = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(run_day, day): day for day in days}

        for future in as_completed(futures):
            date_str, ok, msg = future.result()
            finished += 1
            status = "OK " if ok else "ERR"
            print(f"[{status}] {date_str}  ({finished}/{len(days)} done)")
            if msg:
                for line in msg.splitlines():
                    print(f"      {line}")
            if not ok:
                failed.append(date_str)

    _print_summary(days, failed)


def _print_summary(days, failed):
    completed = len(days) - len(failed)
    print(f"\n{'═'*60}")
    print(f"Completed : {completed}/{len(days)}")
    if failed:
        print(f"Failed    : {', '.join(sorted(failed))}")
        print(f"Resume    : python3 script/run_simulation.py --start <next_day>")
    else:
        print("All days completed successfully.")


if __name__ == "__main__":
    main()
