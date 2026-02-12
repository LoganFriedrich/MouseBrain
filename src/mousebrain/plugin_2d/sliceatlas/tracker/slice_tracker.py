"""
slice_tracker.py - Calibration run tracking for BrainSlice pipeline.

Tracks all calibration runs (registration, detection, colocalization, quantification)
to a CSV file for reproducibility and history.

Usage:
    from mousebrain.plugin_2d.sliceatlas.tracker import SliceTracker

    tracker = SliceTracker()
    run_id = tracker.log_detection(sample_id="ENCR_001_slice12", ...)
    tracker.update_status(run_id, status="completed", det_nuclei_found=1234)
    tracker.mark_as_best(run_id)
"""

import csv
import os
import socket
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

from .schema import CSV_COLUMNS, RUN_TYPES


class SliceTracker:
    """
    Central calibration run tracking for BrainSlice pipeline.

    All runs are logged to a single CSV file with consistent columns.
    """

    def __init__(self, csv_path: Optional[Path] = None):
        """
        Initialize tracker.

        Args:
            csv_path: Path to calibration runs CSV. Uses default if not specified.
        """
        if csv_path is None:
            from ..core.config import TRACKER_CSV
            csv_path = TRACKER_CSV

        self.csv_path = Path(csv_path)
        self._ensure_csv_exists()

    def _ensure_csv_exists(self):
        """Create CSV with headers if it doesn't exist."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()

    def _generate_run_id(self, run_type: str, sample_id: str) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{run_type}_{sample_id}_{timestamp}_{os.getpid()}"
        short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:6]
        return f"{run_type[:3]}_{timestamp}_{short_hash}"

    def _get_hostname(self) -> str:
        """Get current hostname."""
        try:
            return socket.gethostname()
        except:
            return "unknown"

    def _add_hierarchy_fields(self, row: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
        """Auto-populate hierarchy fields from sample name."""
        from ..core.config import parse_sample_name

        parsed = parse_sample_name(sample_id)
        row["project"] = parsed.get("project", "")
        row["cohort"] = parsed.get("cohort", "")
        row["slice_num"] = parsed.get("slice_num", "")

        return row

    def _write_row(self, row: Dict[str, Any]):
        """Append a row to the CSV."""
        full_row = {col: row.get(col, "") for col in CSV_COLUMNS}

        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow(full_row)

    def _read_all(self) -> List[Dict[str, str]]:
        """Read all runs from CSV."""
        if not self.csv_path.exists():
            return []

        with open(self.csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _update_row(self, run_id: str, updates: Dict[str, Any]) -> bool:
        """Update a specific row by run_id."""
        rows = self._read_all()
        found = False

        for row in rows:
            if row['run_id'] == run_id:
                row.update({k: str(v) if v is not None else "" for k, v in updates.items()})
                found = True
                break

        if found:
            clean_rows = []
            for row in rows:
                clean_row = {k: row.get(k, "") for k in CSV_COLUMNS}
                clean_rows.append(clean_row)

            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(clean_rows)

        return found

    # =========================================================================
    # LOGGING METHODS
    # =========================================================================

    def log_registration(
        self,
        sample_id: str,
        atlas: str = "allen_mouse_10um",
        orientation: str = "coronal",
        method: str = "abba_manual",
        ap_position: Optional[float] = None,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = "started",
        script_version: str = "1.0.0",
    ) -> str:
        """Log a slice registration run."""
        run_id = self._generate_run_id("registration", sample_id)

        row = {
            "run_id": run_id,
            "run_type": "registration",
            "sample_id": sample_id,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "reg_atlas": atlas,
            "reg_orientation": orientation,
            "reg_method": method,
            "reg_ap_position": ap_position,
            "input_path": input_path,
            "output_path": output_path,
            "notes": notes,
            "script_version": script_version,
            "hostname": self._get_hostname(),
        }

        row = self._add_hierarchy_fields(row, sample_id)
        self._write_row(row)
        return run_id

    def log_detection(
        self,
        sample_id: str,
        channel: str = "red",
        model: str = "2D_versatile_fluo",
        prob_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        scale: float = 1.0,
        min_area: int = 50,
        max_area: int = 5000,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = "started",
        script_version: str = "1.0.0",
    ) -> str:
        """Log a nuclei detection run."""
        run_id = self._generate_run_id("detection", sample_id)

        row = {
            "run_id": run_id,
            "run_type": "detection",
            "sample_id": sample_id,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "det_channel": channel,
            "det_model": model,
            "det_prob_thresh": prob_thresh,
            "det_nms_thresh": nms_thresh,
            "det_scale": scale,
            "det_min_area": min_area,
            "det_max_area": max_area,
            "input_path": input_path,
            "output_path": output_path,
            "notes": notes,
            "script_version": script_version,
            "hostname": self._get_hostname(),
        }

        row = self._add_hierarchy_fields(row, sample_id)
        self._write_row(row)
        return run_id

    def log_colocalization(
        self,
        sample_id: str,
        signal_channel: str = "green",
        background_method: str = "percentile",
        background_percentile: float = 10.0,
        background_value: Optional[float] = None,
        threshold_method: str = "fold_change",
        threshold_value: float = 2.0,
        parent_run: Optional[str] = None,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = "started",
        script_version: str = "1.0.0",
    ) -> str:
        """Log a colocalization analysis run."""
        run_id = self._generate_run_id("colocalization", sample_id)

        row = {
            "run_id": run_id,
            "run_type": "colocalization",
            "sample_id": sample_id,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "coloc_signal_channel": signal_channel,
            "coloc_background_method": background_method,
            "coloc_background_percentile": background_percentile,
            "coloc_background_value": background_value,
            "coloc_threshold_method": threshold_method,
            "coloc_threshold_value": threshold_value,
            "parent_run": parent_run,
            "input_path": input_path,
            "output_path": output_path,
            "notes": notes,
            "script_version": script_version,
            "hostname": self._get_hostname(),
        }

        row = self._add_hierarchy_fields(row, sample_id)
        self._write_row(row)
        return run_id

    def log_quantification(
        self,
        sample_id: str,
        parent_run: Optional[str] = None,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = "started",
        script_version: str = "1.0.0",
    ) -> str:
        """Log a quantification run."""
        run_id = self._generate_run_id("quantification", sample_id)

        row = {
            "run_id": run_id,
            "run_type": "quantification",
            "sample_id": sample_id,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "parent_run": parent_run,
            "input_path": input_path,
            "output_path": output_path,
            "notes": notes,
            "script_version": script_version,
            "hostname": self._get_hostname(),
        }

        row = self._add_hierarchy_fields(row, sample_id)
        self._write_row(row)
        return run_id

    # =========================================================================
    # UPDATE METHODS
    # =========================================================================

    def update_status(
        self,
        run_id: str,
        status: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        **kwargs
    ) -> bool:
        """
        Update run status and results.

        Common kwargs:
            det_nuclei_found: int - Detection nuclei count
            coloc_positive_cells: int - Positive cells count
            coloc_negative_cells: int - Negative cells count
            coloc_positive_fraction: float - Fraction positive
            quant_total_regions: int - Number of regions with cells
        """
        updates = {}

        if status:
            updates['status'] = status
        if duration_seconds is not None:
            updates['duration_seconds'] = duration_seconds

        # Map common result fields
        result_fields = [
            'det_nuclei_found',
            'coloc_positive_cells', 'coloc_negative_cells', 'coloc_positive_fraction',
            'coloc_background_value',
            'quant_total_regions', 'quant_top_region', 'quant_top_region_count',
            'output_path', 'labels_path', 'measurements_path', 'region_counts_path',
            'reg_transform_path', 'reg_quality_score',
        ]
        for field in result_fields:
            if field in kwargs and kwargs[field] is not None:
                updates[field] = kwargs[field]

        return self._update_row(run_id, updates)

    def rate_run(self, run_id: str, rating: int, notes: Optional[str] = None) -> bool:
        """Rate a run (1-5 stars) with optional notes."""
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be 1-5")

        updates = {'rating': rating}
        if notes:
            rows = self._read_all()
            for row in rows:
                if row['run_id'] == run_id:
                    existing = row.get('notes', '')
                    if existing:
                        updates['notes'] = f"{existing}; {notes}"
                    else:
                        updates['notes'] = notes
                    break

        return self._update_row(run_id, updates)

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def get_run(self, run_id: str) -> Optional[Dict[str, str]]:
        """Get a single run by ID."""
        for row in self._read_all():
            if row['run_id'] == run_id:
                return row
        return None

    def search(
        self,
        sample_id: Optional[str] = None,
        run_type: Optional[str] = None,
        status: Optional[str] = None,
        rating_min: Optional[int] = None,
        limit: int = 50,
        sort_by: str = "created_at",
        descending: bool = True,
    ) -> List[Dict[str, str]]:
        """Search runs with filters."""
        rows = self._read_all()

        if sample_id:
            rows = [r for r in rows if sample_id.lower() in r.get('sample_id', '').lower()]
        if run_type:
            rows = [r for r in rows if r.get('run_type') == run_type]
        if status:
            rows = [r for r in rows if r.get('status') == status]
        if rating_min:
            rows = [r for r in rows if r.get('rating') and int(r['rating']) >= rating_min]

        def sort_key(r):
            val = r.get(sort_by, '')
            if sort_by in ['rating', 'duration_seconds', 'det_nuclei_found']:
                try:
                    return float(val) if val else 0
                except:
                    return 0
            return val

        rows.sort(key=sort_key, reverse=descending)
        return rows[:limit]

    def get_recent(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get most recent runs."""
        return self.search(limit=limit, sort_by="created_at", descending=True)

    def get_best_for_sample(
        self, sample_id: str, run_type: str = 'detection'
    ) -> Optional[Dict[str, str]]:
        """Get the marked-best run for a specific sample."""
        rows = self._read_all()
        for row in rows:
            if (row.get('sample_id') == sample_id and
                row.get('run_type') == run_type and
                row.get('marked_best') == 'True'):
                return row

        # Fall back to highest-rated
        candidates = [r for r in rows if
                      r.get('sample_id') == sample_id and
                      r.get('run_type') == run_type and
                      r.get('rating')]
        if candidates:
            candidates.sort(key=lambda x: int(x.get('rating', 0)), reverse=True)
            return candidates[0]

        return None

    # =========================================================================
    # MARK-AS-BEST
    # =========================================================================

    def mark_as_best(self, run_id: str) -> bool:
        """
        Mark a run as the 'best' for its sample.

        Automatically clears marked_best from previous best for same sample+type.
        """
        run = self.get_run(run_id)
        if not run:
            return False

        sample_id = run.get('sample_id', '')
        run_type = run.get('run_type', '')

        # Clear previous best for this sample+type
        rows = self._read_all()
        for row in rows:
            if (row.get('sample_id') == sample_id and
                row.get('run_type') == run_type and
                row.get('marked_best') == 'True' and
                row['run_id'] != run_id):
                self._update_row(row['run_id'], {'marked_best': ''})

        # Mark this one as best
        return self._update_row(run_id, {
            'marked_best': 'True',
            'rating': 5,
        })

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics."""
        rows = self._read_all()

        stats = {
            'total': len(rows),
            'by_type': {},
            'by_status': {},
            'by_sample': {},
            'rated': 0,
            'avg_rating': 0,
        }

        ratings = []

        for row in rows:
            run_type = row.get('run_type', 'unknown')
            status = row.get('status', 'unknown')
            sample = row.get('sample_id', 'unknown')
            rating = row.get('rating', '')

            stats['by_type'][run_type] = stats['by_type'].get(run_type, 0) + 1
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
            stats['by_sample'][sample] = stats['by_sample'].get(sample, 0) + 1

            if rating:
                stats['rated'] += 1
                ratings.append(int(rating))

        if ratings:
            stats['avg_rating'] = sum(ratings) / len(ratings)

        return stats
