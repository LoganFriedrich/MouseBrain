#!/usr/bin/env python3
"""
util_mark_paradigm_classifier.py

Settle the classification method of an imaging setup: mark ONE classification
run (plain or selective) Paradigm Best. Step 5 then classifies every brain of
that imaging setup with exactly that run's method -- its model file, cube and
batch size, and (selective) its tracing type -- and refuses to run for an
imaging setup that has none.

Usage:
    # Look first: what the run records, what is settled now, what would change.
    python util_mark_paradigm_classifier.py --run sel_20000101_000000_abc123 --dry-run

    # Mark it. A run made before the tracker recorded cube and batch size gets
    # them filled in, with the reason in its notes:
    python util_mark_paradigm_classifier.py --run sel_20000101_000000_abc123 \\
        --cube-size 50 --batch-size 32 \\
        --why "confirmed by <who> on <date>; cube/batch = step 5 defaults the run used"

WHAT IT WRITES
One locked rewrite of the calibration-runs tracker (the tracker's own write
path, ExperimentTracker.mark_paradigm_best): the run gets paradigm_best and
marked_best; the imaging setup's previous Paradigm Best classification (plain
or selective) loses that mark; --cube-size / --batch-size go into the run's
row ONLY where the row has no value (a value the run recorded is never
replaced: if it differs, nothing is written); --why is added to its notes.
Nothing else is changed, no file is moved, nothing is classified.

WHY a run and not "a model": a model file alone is not a method. The same
model gave very different counts with and without selective classification,
and the pre-filter and settings change the result too. A run is the whole
method as it was actually used.

--dry-run reads the tracker file directly and writes nothing at all.
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

CLASSIFYING = ("classification", "selective_classification")


def _read_rows(csv_path: Path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _describe(row: dict) -> list:
    model = row.get("class_model_path") or row.get("model_path") or ""
    lines = [
        f"  run:            {row.get('exp_id')}",
        f"  kind:           {row.get('exp_type')}",
        f"  brain:          {row.get('brain')}",
        f"  made:           {row.get('created_at')}",
        f"  status:         {row.get('status')}",
        f"  imaging setup:  {row.get('imaging_paradigm')}",
        f"  model file:     {model}",
    ]
    try:
        st = Path(model).stat()
        lines.append(f"                  present, {st.st_size:,} bytes, modified "
                     f"{datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')}")
    except OSError:
        lines.append("                  [!] NOT FOUND -- step 5 would refuse to run")
    lines += [
        f"  cube size:      {row.get('class_cube_size') or '(not recorded)'}",
        f"  batch size:     {row.get('class_batch_size') or '(not recorded)'}",
    ]
    if row.get("exp_type") == "selective_classification":
        lines.append(f"  tracing type:   {row.get('prefilter_tracing_type') or '(not recorded)'}")
    lines.append(f"  cells found:    {row.get('class_cells_found')}")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description="Mark a classification run as the settled "
                                             "(Paradigm Best) method of its imaging setup")
    ap.add_argument("--run", required=True, help="exp_id of the classification run")
    ap.add_argument("--cube-size", type=int, help="fill in if the run's row has none")
    ap.add_argument("--batch-size", type=int, help="fill in if the run's row has none")
    ap.add_argument("--why", default="", help="why it is settled; added to the run's notes")
    ap.add_argument("--dry-run", action="store_true", help="show, write nothing")
    ap.add_argument("--tracker", type=Path, help="the tracker CSV (default: the configured one)")
    args = ap.parse_args(argv)

    if args.tracker is None:
        from mousebrain.tracker import DEFAULT_TRACKER_PATH
        args.tracker = Path(DEFAULT_TRACKER_PATH)
    rows = _read_rows(args.tracker)
    row = next((r for r in rows if r.get("exp_id") == args.run), None)
    if row is None:
        print(f"[FAIL] No run {args.run} in {args.tracker}")
        return 1

    print(f"Tracker: {args.tracker}")
    print("The run to settle:")
    print("\n".join(_describe(row)))

    problems = []
    if row.get("exp_type") not in CLASSIFYING:
        problems.append(f"it is a {row.get('exp_type')} run, not a classification")
    if row.get("status") != "completed":
        problems.append(f"its status is '{row.get('status')}', not completed")
    if (row.get("archived_at") or "").strip():
        problems.append("it is archived (superseded)")
    if not row.get("imaging_paradigm"):
        problems.append("it has no imaging setup")
    model = row.get("class_model_path") or row.get("model_path") or ""
    if not model or not Path(model).is_file():
        problems.append("its model file is not there")
    fill = {}
    for col, value, name in (("class_cube_size", args.cube_size, "--cube-size"),
                             ("class_batch_size", args.batch_size, "--batch-size")):
        have = (row.get(col) or "").strip()
        if value is None:
            if not have:
                problems.append(f"it records no {col}; give {name}")
            continue
        if have and int(float(have)) != value:
            problems.append(f"it records {col}={have}; {name} {value} would contradict it")
        elif not have:
            fill[col] = value
    if row.get("exp_type") == "selective_classification" and not row.get("prefilter_tracing_type"):
        problems.append("it records no tracing type")

    paradigm = row.get("imaging_paradigm")
    current = [r for r in rows if r.get("imaging_paradigm") == paradigm
               and r.get("exp_type") in CLASSIFYING and r.get("paradigm_best") == "True"
               and r.get("exp_id") != args.run]
    print(f"\nSettled now for '{paradigm}': "
          + (", ".join(r["exp_id"] for r in current) if current else "nothing"))
    if current:
        print("  (that mark would be cleared)")
    if fill:
        print("Would fill in (empty in the run's row): "
              + ", ".join(f"{k}={v}" for k, v in fill.items()))
    note = "; ".join(filter(None, [
        f"{datetime.now().date().isoformat()} marked Paradigm Best (settled classification "
        f"method) with util_mark_paradigm_classifier",
        args.why.strip(),
        ("filled in: " + ", ".join(f"{k}={v}" for k, v in fill.items())) if fill else ""]))
    print(f"Note to add: {note}")

    if problems:
        print("\n[FAIL] Not marked:")
        for p in problems:
            print(f"  - {p}")
        return 1
    if args.dry_run:
        print("\n[DRY RUN] Nothing was written.")
        return 0

    from mousebrain.tracker import ExperimentTracker
    tracker = ExperimentTracker(args.tracker)
    if not tracker.mark_paradigm_best(args.run, fill_empty=fill, note=note):
        print("[FAIL] The tracker refused the mark (reason above); nothing was written.")
        return 1
    method = tracker.get_paradigm_classification(paradigm)
    print(f"\n[OK] {args.run} is the settled classification method for '{paradigm}':")
    for k in ("method", "model_path", "cube_size", "batch_size", "tracing_type", "missing"):
        print(f"  {k}: {method.get(k)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
