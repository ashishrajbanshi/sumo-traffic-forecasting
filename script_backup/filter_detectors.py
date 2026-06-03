#!/usr/bin/env python3
"""
filter_detectors.py — Remove E1 detectors whose speed variance is too low.

Background (DeepSUMO paper, Vogel 2023, Section 6.2, Figures 19–20)
────────────────────────────────────────────────────────────────────
Detectors on low-traffic or speed-limit-constant roads measure the same
speed at every interval (≈ reference speed, no vehicles present).  When
a graph neural network is trained on data that includes these "flat"
sensors, the model severely underfits on the sensors that do show
meaningful speed variation.  Removing sensors with speed variance below
a threshold (the paper uses 1.0) fixes this.

Workflow
────────
  1.  Run SUMO once (a short test run, e.g. 3 600 s is enough) with
      detectors.add.xml included as an additional file.
  2.  Run this script:
          python script/filter_detectors.py
  3.  A pruned file  detectors_filtered.add.xml  is written.
  4.  Update your .sumocfg additional-files to use the filtered file for
      all subsequent (long) training runs.

Speed correction (paper Section 5.4.2.2, Equation 5)
─────────────────────────────────────────────────────
Intervals where no vehicle passed (SUMO reports speed = -1) are excluded
from the variance calculation.  The paper's DeepSUMO framework corrects
such intervals to the reference (speed-limit) speed at runtime; here we
simply skip them to avoid artificially inflating variance.
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR  = os.path.normpath(os.path.join(BASE_DIR, ".."))

DET_DIR = os.path.join(SIM_DIR, "detectors")

DAY_TYPE_DEFAULTS = {
    "weekday": {
        "detector_output": os.path.join(DET_DIR, "detector_output_weekday.xml"),
        "detectors_file":  os.path.join(DET_DIR, "detectors_weekday.add.xml"),
        "output_file":     os.path.join(DET_DIR, "detectors_filtered_weekday.add.xml"),
    },
    "weekend": {
        "detector_output": os.path.join(DET_DIR, "detector_output_weekend.xml"),
        "detectors_file":  os.path.join(DET_DIR, "detectors_weekend.add.xml"),
        "output_file":     os.path.join(DET_DIR, "detectors_filtered_weekend.add.xml"),
    },
}

# Paper Section 6.2: remove detectors with variance < 1.0
DEFAULT_THRESHOLD = 1.0


# ── Speed data reader ─────────────────────────────────────────────────────────

def read_speeds(xml_path):
    """
    Parse detector_output.xml and return {det_id: [v0, v1, …]} (m/s).
    Intervals without a vehicle (speed == -1) are dropped.
    """
    speeds = defaultdict(list)
    try:
        tree = ET.parse(xml_path)
    except FileNotFoundError:
        print(f"ERROR: {xml_path} not found.")
        print("Run SUMO first so that detector_output.xml is generated.")
        sys.exit(1)
    except ET.ParseError as e:
        print(f"ERROR: Cannot parse {xml_path}: {e}")
        sys.exit(1)

    for interval in tree.getroot().iter("interval"):
        det_id = interval.get("id")
        speed  = interval.get("speed", "-1")
        if det_id is None:
            continue
        try:
            v = float(speed)
            if v >= 0.0:
                speeds[det_id].append(v)
        except ValueError:
            pass

    return dict(speeds)


# ── Variance ──────────────────────────────────────────────────────────────────

def _var(values):
    """Population variance (no numpy dependency)."""
    n = len(values)
    if n < 2:
        return 0.0
    mu  = sum(values) / n
    return sum((x - mu) ** 2 for x in values) / n


# ── Filter ────────────────────────────────────────────────────────────────────

def filter_definitions(det_defs, speeds, threshold, out_path):
    """
    Read det_defs, remove every e1Detector whose speed variance is below
    threshold, write the result to out_path.
    Returns (total, kept, removed).
    """
    try:
        tree = ET.parse(det_defs)
    except FileNotFoundError:
        print(f"ERROR: {det_defs} not found.  Run generate_detectors.py first.")
        sys.exit(1)

    root  = tree.getroot()
    total = removed = 0
    low_variance = []

    for elem in list(root):
        if elem.tag != "e1Detector":
            continue
        total += 1
        det_id = elem.get("id", "")
        var    = _var(speeds.get(det_id, []))
        if var < threshold:
            root.remove(elem)
            removed += 1
            low_variance.append((det_id, var))

    kept = total - removed

    # Print a sample of what was removed for transparency
    if low_variance:
        sample = sorted(low_variance, key=lambda x: x[1])[:15]
        print("  Sample removed (lowest variance):")
        for det_id, var in sample:
            print(f"    {det_id:<55s}  var={var:.4f}")
        if len(low_variance) > 15:
            print(f"    … and {len(low_variance) - 15} more")

    _write_filtered_xml(root, out_path, threshold, kept)
    return total, kept, removed


def _write_filtered_xml(root, path, threshold, kept):
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<!-- E1 detectors filtered by filter_detectors.py",
        f"     Speed-variance threshold : {threshold}",
        f"     Detectors kept           : {kept}",
        '-->',
        '<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '            xsi:noNamespaceSchemaLocation='
        '"http://sumo.dlr.de/xsd/additional_file.xsd">',
    ]
    for elem in root:
        if elem.tag == "e1Detector":
            attrs = " ".join(f'{k}="{v}"' for k, v in elem.attrib.items())
            lines.append(f"    <e1Detector {attrs}/>")
    lines.append("</additional>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── Statistics helper ─────────────────────────────────────────────────────────

def print_variance_distribution(speeds, thresholds=(0.5, 1.0, 2.0, 5.0)):
    """Show how many detectors survive at various threshold choices."""
    total = len(speeds)
    if total == 0:
        return
    print("  Variance distribution (to help choose threshold):")
    print(f"  {'Threshold':>12s}  {'Kept':>6s}  {'Removed':>8s}  {'% kept':>8s}")
    for t in thresholds:
        kept = sum(1 for v in speeds.values() if _var(v) >= t)
        print(f"  {t:>12.1f}  {kept:>6d}  {total - kept:>8d}  {100*kept/total:>7.1f}%")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--day-type", choices=["weekday", "weekend"], default="weekday",
                    help="Which simulation output to filter (default: weekday)")
    ap.add_argument("--detector-output", metavar="XML", default=None,
                    help="Override detector output XML from SUMO run")
    ap.add_argument("--detectors-file",  metavar="XML", default=None,
                    help="Override detector definition file to filter")
    ap.add_argument("--output-file",     metavar="XML", default=None,
                    help="Override destination for filtered definitions")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"Speed-variance threshold; detectors below this are "
                         f"removed (default: {DEFAULT_THRESHOLD}, per paper)")
    ap.add_argument("--show-distribution", action="store_true",
                    help="Print how many detectors survive at several thresholds")
    args = ap.parse_args()

    defaults = DAY_TYPE_DEFAULTS[args.day_type]
    if args.detector_output is None:
        args.detector_output = defaults["detector_output"]
    if args.detectors_file is None:
        args.detectors_file = defaults["detectors_file"]
    if args.output_file is None:
        args.output_file = defaults["output_file"]

    print(f"Reading detector output: {args.detector_output}")
    speeds = read_speeds(args.detector_output)
    print(f"  Intervals found for {len(speeds)} detectors")
    print()

    if args.show_distribution or True:   # always show — helps users pick threshold
        print_variance_distribution(speeds)

    print(f"Filtering  (threshold = {args.threshold}) …")
    total, kept, removed = filter_definitions(
        args.detectors_file, speeds, args.threshold, args.output_file
    )
    print()
    print(f"  Total detectors  : {total}")
    print(f"  Removed          : {removed}  (variance < {args.threshold})")
    print(f"  Kept             : {kept}")
    print()
    print(f"Written → {args.output_file}")
    print()

    _patch_sumocfg(args.day_type)


SUMOCFG_FOR_DAY = {
    "weekday": os.path.join(SIM_DIR, "config", "sumo_chattanooga_weekdays.sumocfg"),
    "weekend": os.path.join(SIM_DIR, "config", "sumo_chattanooga.sumocfg"),
}


def _patch_sumocfg(day_type):
    """Replace the unfiltered detector file reference with the filtered one."""
    cfg_path  = SUMOCFG_FOR_DAY[day_type]
    unfiltered = f"detectors_{day_type}.add.xml"
    filtered   = f"detectors_filtered_{day_type}.add.xml"

    with open(cfg_path, "r", encoding="utf-8") as f:
        content = f.read()

    if unfiltered not in content:
        print(f"Config already uses filtered file: {cfg_path}")
        return

    updated = content.replace(unfiltered, filtered)
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(updated)

    print(f"Updated {cfg_path}")
    print(f"  {unfiltered}  →  {filtered}")
    print()
    print(f"Ready for full run:")
    if day_type == "weekday":
        print("  sumo -c config/sumo_chattanooga_weekdays.sumocfg")
    else:
        print("  sumo -c config/sumo_chattanooga.sumocfg")


if __name__ == "__main__":
    main()
