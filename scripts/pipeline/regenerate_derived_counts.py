#!/usr/bin/env python3
"""
regenerate_derived_counts.py - Rebuild the derived summary tables from region_counts.csv.

WHY THIS EXISTS
---------------
region_counts.csv (one row per brain, one column per Allen region) is the
primary product of step 6. Two tables are DERIVED from it and must never drift
from it:

  elife_counts.csv                   the 25 eLife tract groups per brain
                                     (+ per-hemisphere columns when present)
  hemisphere_laterality_analysis.csv left/right totals and a laterality index
                                     per eLife group per brain

Step 6 updates elife_counts.csv itself after every brain, but for months its
eLife import failed silently (the mapping module was not on its path), so a
stand-alone regeneration script lived NEXT TO THE DATA with a hardcoded lab
path. This is that script's function, moved into the tool: paths come from
mousebrain.config, the mapping from mousebrain.region_mapping, and it can be
run any time to bring the derived tables back in line with region_counts.csv.
Integrators (e.g. a lab database pulling eLife counts) read elife_counts.csv,
so keeping it current is part of the tool's job.

Old rows of elife_counts.csv are appended to elife_counts_archive.csv with an
archived_at stamp before the file is rewritten (the lab archives, never deletes).

Usage:
    python regenerate_derived_counts.py            # rebuild both tables
    python regenerate_derived_counts.py --dry-run  # report what would be written
    python regenerate_derived_counts.py --summary-dir <folder>   # another 2_Data_Summary
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

FIXED_COLUMNS = ['brain', 'run_date', 'brain_id', 'subject', 'cohort', 'project_code', 'project_name',
                 'det_preset', 'det_ball_xy', 'det_ball_z', 'det_soma_diameter', 'det_threshold',
                 'atlas', 'voxel_xy', 'voxel_z', 'total_cells', 'total_left', 'total_right']


def _group_col(prefix: str, group_name: str) -> str:
    return prefix + group_name.replace(' ', '_').replace('/', '_')


def _int(v) -> int:
    try:
        return int(v) if str(v).strip() else 0
    except (ValueError, TypeError):
        return 0


def extract_region_counts(row: Dict[str, str]) -> Dict[str, int]:
    """{acronym: count} from the region_<acr> columns (totals only)."""
    out = {}
    for k, v in row.items():
        if k.startswith('region_') and not k.startswith('region_left_') and not k.startswith('region_right_'):
            c = _int(v)
            if c > 0:
                out[k[len('region_'):]] = c
    return out


def extract_hemisphere_counts(row: Dict[str, str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    left, right = {}, {}
    for k, v in row.items():
        c = _int(v)
        if c <= 0:
            continue
        if k.startswith('region_left_'):
            left[k[len('region_left_'):]] = c
        elif k.startswith('region_right_'):
            right[k[len('region_right_'):]] = c
    return left, right


def _count(agg: dict, group: str) -> int:
    info = agg.get(group)
    return info['count'] if isinstance(info, dict) else 0


def build_elife_rows(rows: List[Dict[str, str]], aggregate_to_elife, elife_groups) -> Tuple[List[dict], List[str]]:
    """The elife_counts.csv rows and their column order, from region_counts rows."""
    new_rows, all_columns = [], set()
    for row in rows:
        brain_name = row.get('brain', '')
        region_counts = extract_region_counts(row)
        left_counts, right_counts = extract_hemisphere_counts(row)
        elife_total = aggregate_to_elife(region_counts)
        elife_left = aggregate_to_elife(left_counts) if left_counts else {}
        elife_right = aggregate_to_elife(right_counts) if right_counts else {}

        new_row = {c: row.get(c, '') for c in FIXED_COLUMNS}
        new_row['brain'] = brain_name
        new_row['brain_id'] = brain_name.split('/')[0].split('_')[0]
        new_row['atlas'] = row.get('atlas', 'allen_mouse_10um')
        new_row['total_cells'] = _int(row.get('total_cells', 0))

        for group_name in list(elife_groups) + ['[Unmapped]', 'Unused']:
            label = 'Unmapped' if group_name == '[Unmapped]' else group_name
            new_row[_group_col('group_', label)] = _count(elife_total, group_name)
            if left_counts:
                new_row[_group_col('group_left_', label)] = _count(elife_left, group_name)
            if right_counts:
                new_row[_group_col('group_right_', label)] = _count(elife_right, group_name)
        all_columns.update(new_row.keys())
        new_rows.append(new_row)

    group_cols = sorted(c for c in all_columns if c.startswith('group_')
                        and not c.startswith('group_left_') and not c.startswith('group_right_'))
    left_cols = sorted(c for c in all_columns if c.startswith('group_left_'))
    right_cols = sorted(c for c in all_columns if c.startswith('group_right_'))
    return new_rows, FIXED_COLUMNS + group_cols + left_cols + right_cols


def build_laterality_rows(rows: List[Dict[str, str]], aggregate_to_elife, elife_groups) -> List[dict]:
    results = []
    for row in rows:
        brain_name = row.get('brain', '')
        short = brain_name.split('/')[0].split('_')[0]
        total_left, total_right = _int(row.get('total_left', 0)), _int(row.get('total_right', 0))
        if total_left + total_right == 0:
            continue
        left_counts, right_counts = extract_hemisphere_counts(row)
        elife_left = aggregate_to_elife(left_counts) if left_counts else {}
        elife_right = aggregate_to_elife(right_counts) if right_counts else {}
        for group_name in elife_groups:
            l, r = _count(elife_left, group_name), _count(elife_right, group_name)
            ratio = (l / r) if r > 0 else (float('inf') if l > 0 else 0)
            results.append({
                'brain_id': short, 'brain': brain_name, 'elife_group': group_name,
                'left': l, 'right': r, 'total': l + r,
                'lr_ratio': round(ratio, 3) if ratio != float('inf') else 'inf',
                'laterality_index': round((l - r) / (l + r), 3) if (l + r) > 0 else 0,
                'dominant': 'L' if l > r else ('R' if r > l else 'equal'),
            })
    return results


def archive_existing(csv_path: Path, archive_path: Path) -> int:
    """Append the current rows of csv_path to archive_path with an archived_at stamp."""
    if not csv_path.exists():
        return 0
    with open(csv_path, newline='', encoding='utf-8') as f:
        existing = list(csv.DictReader(f))
    if not existing:
        return 0
    stamp = datetime.now().isoformat()
    write_header = not archive_path.exists()
    with open(archive_path, 'a', newline='', encoding='utf-8') as f:
        for old in existing:
            old['archived_at'] = stamp
            writer = csv.DictWriter(f, fieldnames=sorted(old.keys()))
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow(old)
    return len(existing)


def regenerate(summary_dir: Path, dry_run: bool = False, log=print) -> dict:
    from mousebrain.region_mapping import aggregate_to_elife, ELIFE_GROUPS
    region_csv = summary_dir / 'region_counts.csv'
    elife_csv = summary_dir / 'elife_counts.csv'
    elife_archive = summary_dir / 'elife_counts_archive.csv'
    laterality_csv = summary_dir / 'hemisphere_laterality_analysis.csv'
    if not region_csv.exists():
        raise FileNotFoundError("region_counts.csv not found in %s" % summary_dir)
    with open(region_csv, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    elife_rows, columns = build_elife_rows(rows, aggregate_to_elife, ELIFE_GROUPS)
    lat_rows = build_laterality_rows(rows, aggregate_to_elife, ELIFE_GROUPS)
    log("%s%d brains in region_counts.csv -> %d eLife rows (%d columns), %d laterality rows" % (
        "[DRY RUN] " if dry_run else "", len(rows), len(elife_rows), len(columns), len(lat_rows)))
    if dry_run:
        return {"brains": len(rows), "elife_rows": len(elife_rows), "laterality_rows": len(lat_rows), "written": False}
    archived = archive_existing(elife_csv, elife_archive)
    with open(elife_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in elife_rows:
            writer.writerow({k: r.get(k, '') for k in columns})
    if lat_rows:
        cols = ['brain_id', 'brain', 'elife_group', 'left', 'right', 'total', 'lr_ratio', 'laterality_index', 'dominant']
        with open(laterality_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(lat_rows)
    log("[OK] wrote %s (%d rows; %d old rows archived) and %s (%d rows)" % (
        elife_csv.name, len(elife_rows), archived, laterality_csv.name, len(lat_rows)))
    return {"brains": len(rows), "elife_rows": len(elife_rows), "laterality_rows": len(lat_rows),
            "archived": archived, "written": True}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--summary-dir', type=Path, default=None,
                    help='The 2_Data_Summary folder (default: from mousebrain.config)')
    ap.add_argument('--dry-run', action='store_true', help='Report only; write nothing')
    args = ap.parse_args(argv)
    summary_dir = args.summary_dir
    if summary_dir is None:
        try:
            from mousebrain.config import DATA_SUMMARY_DIR
            summary_dir = Path(DATA_SUMMARY_DIR)
        except Exception as e:
            print("[FAIL] cannot resolve the summary folder (%s); set CONNECTOME_ROOT or pass --summary-dir" % e)
            return 1
    try:
        regenerate(summary_dir, dry_run=args.dry_run)
    except Exception as e:
        print("[FAIL] %s" % e)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
