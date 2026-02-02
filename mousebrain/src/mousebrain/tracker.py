#!/usr/bin/env python3
"""
tracker.py (mousebrain package)

Calibration run tracking system for the BrainGlobe pipeline.

This module is now part of the mousebrain package and tracks CALIBRATION runs
(tuning/optimization) - detection parameter tests, model training, classification
tests, etc. These are iterative experiments to find optimal settings.

For PRODUCTION results (final cell counts per region), see region_counts.csv
which is updated by 6_count_regions.py.

Two-file architecture:
  - calibration_runs.csv: This tracker - all tuning runs with parameters
  - region_counts.csv: Production results - one row per brain, wide format

This module is imported by the run_* wrapper scripts and the napari tuning widget.

Usage as a module:
    from mousebrain.tracker import ExperimentTracker

    tracker = ExperimentTracker()
    exp_id = tracker.log_detection(brain="349_CNT_01_02_1p625x_z4", ...)
    tracker.update_status(exp_id, status="completed", cells_found=1234)
    tracker.rate_experiment(exp_id, rating=4, notes="Good results")

Usage as CLI:
    python -m mousebrain.tracker                    # Show recent runs
    python -m mousebrain.tracker --search "349"     # Search by brain
    python -m mousebrain.tracker --stats            # Show statistics
"""

import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
import hashlib
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default location for the calibration runs CSV
from mousebrain.config import CALIBRATION_RUNS_CSV as DEFAULT_TRACKER_PATH, parse_brain_name

# Experiment types
EXP_TYPES = ["detection", "training", "classification", "registration", "counts"]

# CSV columns (order matters for readability)
CSV_COLUMNS = [
    # Identity
    "exp_id",
    "exp_type",
    "brain",
    "created_at",

    # Data Hierarchy (auto-filled from brain name)
    # See 2_Data_Summary/DATA_HIERARCHY.md for documentation
    "brain_id",       # e.g., "349" - individual tissue sample
    "subject",        # e.g., "CNT_01_02" - the animal/unit of study
    "cohort",         # e.g., "CNT_01" - group processed together
    "project_code",   # e.g., "CNT" - umbrella project code
    "project_name",   # e.g., "Connectome" - human-readable project name

    # Status
    "status",  # started, running, completed, failed, cancelled
    "duration_seconds",

    # Common parameters
    "atlas",
    "voxel_z",
    "voxel_xy",

    # Detection parameters
    "det_preset",
    "det_ball_xy",
    "det_ball_z",
    "det_soma_diameter",
    "det_threshold",
    "det_cells_found",
    "det_scope",        # "full" or "partial" - whether full brain or subset was analyzed
    "det_z_start",      # Starting Z slice (for partial detection)
    "det_z_end",        # Ending Z slice (for partial detection)
    "det_z_center",     # Center Z slice (for partial detection)

    # Training parameters
    "train_epochs",
    "train_learning_rate",
    "train_augment",
    "train_pretrained",
    "train_loss",
    "train_accuracy",

    # Classification parameters
    "class_model_path",
    "class_cube_size",
    "class_batch_size",
    "class_cells_found",
    "class_rejected",

    # Counting parameters
    "count_regions",
    "count_total_cells",
    "count_output_csv",

    # Registration parameters
    "reg_atlas",            # Atlas used (e.g., "allen_mouse_10um")
    "reg_orientation",      # Brain orientation (e.g., "prs")
    "reg_voxel_z",          # Voxel size Z in microns
    "reg_voxel_xy",         # Voxel size XY in microns
    "reg_affine_n_steps",   # Number of affine registration steps
    "reg_freeform_n_steps", # Number of freeform registration steps
    "reg_approved",         # True if user approved the registration
    "reg_approved_at",      # ISO timestamp of approval
    "reg_qc_images",        # Paths to QC images (comma-separated)

    # Paths
    "input_path",
    "output_path",
    "model_path",

    # Linkage
    "parent_experiment",  # Links classification to detection, etc.

    # User feedback
    "rating",  # 1-5 stars
    "notes",
    "tags",

    # Metadata
    "script_version",
    "hostname",

    # QC and documentation
    "qc_image_path",  # Path to auto-generated QC image

    # Comparison and archival tracking (added for tuning widget workflow)
    "marked_best",      # True if user marked this as best for its brain
    "archived_by",      # exp_id that replaced/superseded this experiment
    "archived_at",      # ISO timestamp when archived
    "diff_vs_best",     # Total difference points when compared to best
    "diff_gained",      # Points gained vs reference
    "diff_lost",        # Points lost vs reference
    "comparison_ref",   # exp_id this was compared against

    # Imaging paradigm (for auto-applying settings to similar brains)
    "imaging_paradigm",  # Extracted from brain name, e.g., "1p625x_z4"
    "paradigm_best",     # True if this is the best run for its paradigm (not just brain)
]


# =============================================================================
# EXPERIMENT TRACKER CLASS
# =============================================================================

class ExperimentTracker:
    """
    Central experiment tracking for BrainGlobe pipeline.

    All experiments are logged to a single CSV file with consistent columns.
    """

    def __init__(self, csv_path: Optional[Path] = None):
        """
        Initialize tracker.

        Args:
            csv_path: Path to experiments CSV. Uses default if not specified.
        """
        self.csv_path = Path(csv_path) if csv_path else DEFAULT_TRACKER_PATH
        self._ensure_csv_exists()

    def _ensure_csv_exists(self):
        """Create CSV with headers if it doesn't exist."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()

    def _generate_exp_id(self, exp_type: str, brain: str) -> str:
        """Generate a unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Short hash for uniqueness
        hash_input = f"{exp_type}_{brain}_{timestamp}_{os.getpid()}"
        short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:6]
        return f"{exp_type[:3]}_{timestamp}_{short_hash}"

    def _get_hostname(self) -> str:
        """Get current hostname."""
        import socket
        try:
            return socket.gethostname()
        except:
            return "unknown"

    def _add_hierarchy_fields(self, row: Dict[str, Any], brain: str) -> Dict[str, Any]:
        """
        Auto-populate hierarchy fields from brain name.

        Parses brain names like "349_CNT_01_02_1p625x_z4" into:
          - brain_id: 349
          - subject: CNT_01_02
          - cohort: CNT_01
          - project_code: CNT
          - project_name: Connectome
          - imaging_paradigm: 1p625x_z4

        See 2_Data_Summary/DATA_HIERARCHY.md for full documentation.
        """
        parsed = parse_brain_name(brain)

        row["brain_id"] = parsed.get("brain_id", "")
        row["subject"] = parsed.get("subject_full", "")
        row["cohort"] = parsed.get("cohort_full", "")
        row["project_code"] = parsed.get("project_code", "")
        row["project_name"] = parsed.get("project_name", "")
        row["imaging_paradigm"] = parsed.get("imaging_params", "")

        return row

    def _write_row(self, row: Dict[str, Any]):
        """Append a row to the CSV."""
        # Ensure all columns exist with empty string as default
        full_row = {col: row.get(col, "") for col in CSV_COLUMNS}

        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow(full_row)

    def _read_all(self) -> List[Dict[str, str]]:
        """Read all experiments from CSV."""
        if not self.csv_path.exists():
            return []

        with open(self.csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _update_row(self, exp_id: str, updates: Dict[str, Any]) -> bool:
        """Update a specific row by exp_id."""
        rows = self._read_all()
        found = False

        for row in rows:
            if row['exp_id'] == exp_id:
                row.update({k: str(v) if v is not None else "" for k, v in updates.items()})
                found = True
                break

        if found:
            # Filter rows to only include valid CSV_COLUMNS keys
            # This handles cases where CSV has extra/invalid columns
            clean_rows = []
            for row in rows:
                clean_row = {k: row.get(k, "") for k in CSV_COLUMNS}
                clean_rows.append(clean_row)

            # Rewrite entire file
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(clean_rows)

        return found

    # =========================================================================
    # LOGGING METHODS
    # =========================================================================

    def log_detection(
        self,
        brain: str,
        preset: Optional[str] = None,
        ball_xy: Optional[float] = None,
        ball_z: Optional[float] = None,
        soma_diameter: Optional[float] = None,
        threshold: Optional[int] = None,
        voxel_z: Optional[float] = None,
        voxel_xy: Optional[float] = None,
        atlas: str = "allen_mouse_10um",
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Optional[str] = None,
        status: str = "started",
        script_version: str = "1.0.0",
        # Detection scope parameters (for partial vs full detection)
        det_scope: Optional[str] = None,      # "full" or "partial"
        det_z_start: Optional[int] = None,    # Starting Z slice (for partial)
        det_z_end: Optional[int] = None,      # Ending Z slice (for partial)
        det_z_center: Optional[int] = None,   # Center Z slice (for partial)
    ) -> str:
        """
        Log a cell detection experiment.

        Hierarchy fields (brain_id, subject, cohort, project_code, project_name)
        are automatically populated from the brain name.

        Args:
            brain: Brain name
            preset: Detection preset used
            ball_xy/ball_z: Ball filter sizes
            soma_diameter: Expected cell diameter
            threshold: Detection threshold
            voxel_z/voxel_xy: Voxel sizes in microns
            atlas: Atlas used for registration
            input_path: Path to input data
            output_path: Path where results are saved
            notes: User notes
            tags: Comma-separated tags
            status: Run status
            script_version: Version of detection script
            det_scope: "full" for whole brain, "partial" for subset of Z slices
            det_z_start: Starting Z slice for partial detection
            det_z_end: Ending Z slice for partial detection
            det_z_center: Center Z slice for partial detection

        Returns:
            exp_id: Unique experiment identifier
        """
        exp_id = self._generate_exp_id("detection", brain)

        row = {
            "exp_id": exp_id,
            "exp_type": "detection",
            "brain": brain,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "atlas": atlas,
            "voxel_z": voxel_z,
            "voxel_xy": voxel_xy,
            "det_preset": preset,
            "det_ball_xy": ball_xy,
            "det_ball_z": ball_z,
            "det_soma_diameter": soma_diameter,
            "det_threshold": threshold,
            "det_scope": det_scope,
            "det_z_start": det_z_start,
            "det_z_end": det_z_end,
            "det_z_center": det_z_center,
            "input_path": input_path,
            "output_path": output_path,
            "notes": notes,
            "tags": tags,
            "script_version": script_version,
            "hostname": self._get_hostname(),
        }

        # Auto-fill hierarchy from brain name
        row = self._add_hierarchy_fields(row, brain)

        self._write_row(row)
        return exp_id

    def log_training(
        self,
        brain: str,
        epochs: int,
        learning_rate: float = 0.0001,
        augment: bool = True,
        pretrained: Optional[str] = None,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = "started",
        script_version: str = "1.0.0",
    ) -> str:
        """Log a training experiment. Hierarchy fields auto-populated from brain name."""
        exp_id = self._generate_exp_id("training", brain)

        row = {
            "exp_id": exp_id,
            "exp_type": "training",
            "brain": brain,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "train_epochs": epochs,
            "train_learning_rate": learning_rate,
            "train_augment": augment,
            "train_pretrained": pretrained,
            "input_path": input_path,
            "output_path": output_path,
            "notes": notes,
            "script_version": script_version,
            "hostname": self._get_hostname(),
        }

        row = self._add_hierarchy_fields(row, brain)
        self._write_row(row)
        return exp_id

    def log_classification(
        self,
        brain: str,
        model_path: str,
        candidates_path: Optional[str] = None,
        cube_size: int = 50,
        batch_size: int = 32,
        n_free_cpus: int = 2,
        output_path: Optional[str] = None,
        parent_experiment: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = "started",
        script_version: str = "1.0.0",
    ) -> str:
        """Log a classification experiment. Hierarchy fields auto-populated from brain name."""
        exp_id = self._generate_exp_id("classification", brain)

        row = {
            "exp_id": exp_id,
            "exp_type": "classification",
            "brain": brain,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "class_model_path": model_path,
            "class_cube_size": cube_size,
            "class_batch_size": batch_size,
            "input_path": candidates_path,
            "output_path": output_path,
            "model_path": model_path,
            "parent_experiment": parent_experiment,
            "notes": notes,
            "script_version": script_version,
            "hostname": self._get_hostname(),
        }

        row = self._add_hierarchy_fields(row, brain)
        self._write_row(row)
        return exp_id

    def log_counts(
        self,
        brain: str,
        cells_path: str,
        atlas: str = "allen_mouse_10um",
        output_path: Optional[str] = None,
        parent_experiment: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = "started",
        script_version: str = "1.0.0",
    ) -> str:
        """Log a regional counting experiment. Hierarchy fields auto-populated from brain name."""
        exp_id = self._generate_exp_id("counts", brain)

        row = {
            "exp_id": exp_id,
            "exp_type": "counts",
            "brain": brain,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "atlas": atlas,
            "input_path": cells_path,
            "output_path": output_path,
            "parent_experiment": parent_experiment,
            "notes": notes,
            "script_version": script_version,
            "hostname": self._get_hostname(),
        }

        row = self._add_hierarchy_fields(row, brain)
        self._write_row(row)
        return exp_id

    def log_registration(
        self,
        brain: str,
        atlas: str = "allen_mouse_10um",
        orientation: str = "prs",
        voxel_z: Optional[float] = None,
        voxel_xy: Optional[float] = None,
        affine_n_steps: int = 6,
        freeform_n_steps: int = 6,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = "started",
        script_version: str = "1.0.0",
    ) -> str:
        """
        Log a brain registration experiment.

        Tracks atlas registration using BrainGlobe's brainreg.

        Args:
            brain: Brain name
            atlas: Atlas used for registration (e.g., "allen_mouse_10um")
            orientation: Brain orientation string (e.g., "prs" for posterior-right-superior)
            voxel_z: Voxel size in Z (microns)
            voxel_xy: Voxel size in XY (microns)
            affine_n_steps: Number of affine registration steps
            freeform_n_steps: Number of freeform registration steps
            input_path: Path to input data (cropped brain)
            output_path: Path to registration output folder
            notes: User notes
            status: Run status
            script_version: Version of registration script

        Returns:
            exp_id: Unique experiment identifier
        """
        exp_id = self._generate_exp_id("registration", brain)

        row = {
            "exp_id": exp_id,
            "exp_type": "registration",
            "brain": brain,
            "created_at": datetime.now().isoformat(),
            "status": status,
            "atlas": atlas,
            "reg_atlas": atlas,
            "reg_orientation": orientation,
            "reg_voxel_z": voxel_z,
            "reg_voxel_xy": voxel_xy,
            "voxel_z": voxel_z,
            "voxel_xy": voxel_xy,
            "reg_affine_n_steps": affine_n_steps,
            "reg_freeform_n_steps": freeform_n_steps,
            "input_path": input_path,
            "output_path": output_path,
            "notes": notes,
            "script_version": script_version,
            "hostname": self._get_hostname(),
        }

        row = self._add_hierarchy_fields(row, brain)
        self._write_row(row)
        return exp_id

    def approve_registration(self, exp_id: str) -> bool:
        """
        Mark a registration as approved by the user.

        Called after user visually confirms registration quality.

        Args:
            exp_id: The registration experiment ID to approve

        Returns:
            True if successful
        """
        return self._update_row(exp_id, {
            'reg_approved': 'True',
            'reg_approved_at': datetime.now().isoformat(),
            'status': 'approved',
        })

    def add_registration_qc_images(self, exp_id: str, image_paths: List[str]) -> bool:
        """
        Add QC image paths to a registration experiment.

        Args:
            exp_id: The registration experiment ID
            image_paths: List of paths to QC images

        Returns:
            True if successful
        """
        return self._update_row(exp_id, {
            'reg_qc_images': ','.join(str(p) for p in image_paths),
        })

    # =========================================================================
    # UPDATE METHODS
    # =========================================================================

    def update_status(
        self,
        exp_id: str,
        status: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        **kwargs
    ) -> bool:
        """
        Update experiment status and results.

        Common kwargs:
            det_cells_found: int - Detection cell count
            class_cells_found: int - Classification cell count
            class_rejected: int - Rejected candidates
            train_loss: float - Training loss
            train_accuracy: float - Training accuracy
            count_total_cells: int - Total cells counted
        """
        updates = {}

        if status:
            updates['status'] = status
        if duration_seconds is not None:
            updates['duration_seconds'] = duration_seconds

        # Map common result fields
        result_fields = [
            'det_cells_found', 'class_cells_found', 'class_rejected',
            'train_loss', 'train_accuracy', 'count_total_cells',
            'count_output_csv', 'output_path',
            # Comparison fields (for auto-compare after training)
            'diff_vs_best', 'diff_gained', 'diff_lost', 'comparison_ref'
        ]
        for field in result_fields:
            if field in kwargs and kwargs[field] is not None:
                updates[field] = kwargs[field]

        return self._update_row(exp_id, updates)

    def rate_experiment(
        self,
        exp_id: str,
        rating: int,
        notes: Optional[str] = None
    ) -> bool:
        """
        Rate an experiment (1-5 stars) with optional notes.
        """
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be 1-5")

        updates = {'rating': rating}
        if notes:
            # Append to existing notes
            rows = self._read_all()
            for row in rows:
                if row['exp_id'] == exp_id:
                    existing = row.get('notes', '')
                    if existing:
                        updates['notes'] = f"{existing}; {notes}"
                    else:
                        updates['notes'] = notes
                    break

        return self._update_row(exp_id, updates)

    def add_tags(self, exp_id: str, tags: List[str]) -> bool:
        """Add tags to an experiment."""
        rows = self._read_all()
        for row in rows:
            if row['exp_id'] == exp_id:
                existing = row.get('tags', '')
                if existing:
                    existing_tags = set(existing.split(','))
                    existing_tags.update(tags)
                    new_tags = ','.join(sorted(existing_tags))
                else:
                    new_tags = ','.join(sorted(tags))
                return self._update_row(exp_id, {'tags': new_tags})
        return False

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, str]]:
        """Get a single experiment by ID."""
        for row in self._read_all():
            if row['exp_id'] == exp_id:
                return row
        return None

    def search(
        self,
        brain: Optional[str] = None,
        exp_type: Optional[str] = None,
        status: Optional[str] = None,
        rating_min: Optional[int] = None,
        limit: int = 50,
        sort_by: str = "created_at",
        descending: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Search experiments with filters.
        """
        rows = self._read_all()

        # Filter
        if brain:
            rows = [r for r in rows if brain.lower() in r.get('brain', '').lower()]
        if exp_type:
            rows = [r for r in rows if r.get('exp_type') == exp_type]
        if status:
            rows = [r for r in rows if r.get('status') == status]
        if rating_min:
            rows = [r for r in rows if r.get('rating') and int(r['rating']) >= rating_min]

        # Sort
        def sort_key(r):
            val = r.get(sort_by, '')
            # Handle numeric sorting
            if sort_by in ['rating', 'duration_seconds', 'det_cells_found']:
                try:
                    return float(val) if val else 0
                except:
                    return 0
            return val

        rows.sort(key=sort_key, reverse=descending)

        return rows[:limit]

    def get_recent(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get most recent experiments."""
        return self.search(limit=limit, sort_by="created_at", descending=True)

    def get_best(self, exp_type: str, brain: Optional[str] = None, limit: int = 5) -> List[Dict[str, str]]:
        """Get highest-rated experiments of a type."""
        return self.search(
            brain=brain,
            exp_type=exp_type,
            rating_min=1,
            limit=limit,
            sort_by="rating",
            descending=True
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics."""
        rows = self._read_all()

        stats = {
            'total': len(rows),
            'by_type': {},
            'by_status': {},
            'by_brain': {},
            'rated': 0,
            'avg_rating': 0,
        }

        ratings = []

        for row in rows:
            exp_type = row.get('exp_type', 'unknown')
            status = row.get('status', 'unknown')
            brain = row.get('brain', 'unknown')
            rating = row.get('rating', '')

            stats['by_type'][exp_type] = stats['by_type'].get(exp_type, 0) + 1
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
            stats['by_brain'][brain] = stats['by_brain'].get(brain, 0) + 1

            if rating and rating.isdigit():
                stats['rated'] += 1
                ratings.append(int(rating))

        if ratings:
            stats['avg_rating'] = sum(ratings) / len(ratings)

        return stats

    def compare_experiments(self, exp_ids: List[str]) -> List[Dict[str, str]]:
        """Get multiple experiments for comparison."""
        rows = self._read_all()
        return [r for r in rows if r['exp_id'] in exp_ids]

    # =========================================================================
    # MARK-AS-BEST AND ARCHIVAL METHODS
    # =========================================================================

    def mark_as_best(self, exp_id: str) -> bool:
        """
        Mark an experiment as the 'best' for its brain.

        This automatically:
        1. Clears marked_best from any previous best for the same brain
        2. Sets marked_best=True and rating=5 for this experiment

        Args:
            exp_id: The experiment ID to mark as best

        Returns:
            True if successful
        """
        # Get the experiment to find its brain
        exp = self.get_experiment(exp_id)
        if not exp:
            return False

        brain = exp.get('brain', '')
        exp_type = exp.get('exp_type', '')

        # Clear previous best for this brain+type
        rows = self._read_all()
        for row in rows:
            if (row.get('brain') == brain and
                row.get('exp_type') == exp_type and
                row.get('marked_best') == 'True' and
                row['exp_id'] != exp_id):
                # Clear the previous best
                self._update_row(row['exp_id'], {'marked_best': ''})

        # Mark this one as best
        return self._update_row(exp_id, {
            'marked_best': 'True',
            'rating': 5,
        })

    def archive_experiment(self, exp_id: str, archived_by: Optional[str] = None) -> bool:
        """
        Archive an experiment (mark it as superseded).

        Used when a new experiment replaces an old one as "best".

        Args:
            exp_id: The experiment ID to archive
            archived_by: The exp_id that replaced this one (optional)

        Returns:
            True if successful
        """
        return self._update_row(exp_id, {
            'archived_by': archived_by or '',
            'archived_at': datetime.now().isoformat(),
            'marked_best': '',  # Clear best status
        })

    # =========================================================================
    # IMAGING PARADIGM METHODS
    # =========================================================================

    def mark_paradigm_best(self, exp_id: str) -> bool:
        """
        Mark an experiment as best for its imaging paradigm.

        This is different from mark_as_best which is per-brain.
        Paradigm best applies to ALL brains with the same imaging parameters
        (e.g., all 1p625x_z4 brains will use these detection settings).

        Args:
            exp_id: The experiment ID to mark as paradigm best

        Returns:
            True if successful
        """
        exp = self.get_experiment(exp_id)
        if not exp:
            return False

        paradigm = exp.get('imaging_paradigm', '')
        exp_type = exp.get('exp_type', '')

        if not paradigm:
            print(f"[Tracker] Cannot mark paradigm best: no imaging_paradigm for {exp_id}")
            return False

        # Clear previous paradigm_best for this paradigm+type
        rows = self._read_all()
        for row in rows:
            if (row.get('imaging_paradigm') == paradigm and
                row.get('exp_type') == exp_type and
                row.get('paradigm_best') == 'True' and
                row['exp_id'] != exp_id):
                self._update_row(row['exp_id'], {'paradigm_best': ''})

        # Mark this one as paradigm best
        return self._update_row(exp_id, {
            'paradigm_best': 'True',
            'marked_best': 'True',  # Also mark as best for its brain
            'rating': 5,
        })

    def get_best_for_paradigm(
        self,
        imaging_paradigm: str,
        exp_type: str = "detection"
    ) -> Optional[Dict[str, str]]:
        """
        Get the best experiment for an imaging paradigm.

        Use this to auto-apply settings to new brains with the same imaging parameters.

        Args:
            imaging_paradigm: e.g., "1p625x_z4"
            exp_type: "detection", "classification", or "training"

        Returns:
            The paradigm-best experiment dict, or None
        """
        rows = self._read_all()

        # Look for paradigm_best first
        for row in rows:
            if (row.get('imaging_paradigm') == imaging_paradigm and
                row.get('exp_type') == exp_type and
                row.get('paradigm_best') == 'True'):
                return row

        # Fallback: find any marked_best for this paradigm
        candidates = [
            r for r in rows
            if (r.get('imaging_paradigm') == imaging_paradigm and
                r.get('exp_type') == exp_type and
                r.get('marked_best') == 'True')
        ]

        if candidates:
            # Return most recent
            candidates.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            return candidates[0]

        return None

    def get_paradigm_detection_settings(self, imaging_paradigm: str) -> Optional[Dict[str, Any]]:
        """
        Get recommended detection settings for an imaging paradigm.

        Returns a dict with detection parameters that worked well for
        other brains with the same imaging paradigm.

        Args:
            imaging_paradigm: e.g., "1p625x_z4"

        Returns:
            Dict with keys: ball_xy, ball_z, soma_diameter, threshold, preset
            Or None if no paradigm best found
        """
        best = self.get_best_for_paradigm(imaging_paradigm, exp_type="detection")
        if not best:
            return None

        return {
            'ball_xy': float(best.get('det_ball_xy') or 6),
            'ball_z': float(best.get('det_ball_z') or 15),
            'soma_diameter': float(best.get('det_soma_diameter') or 16),
            'threshold': int(best.get('det_threshold') or 10),
            'preset': best.get('det_preset') or 'custom',
            'source_exp_id': best.get('exp_id'),
            'source_brain': best.get('brain'),
        }

    def get_paradigm_model(self, imaging_paradigm: str) -> Optional[str]:
        """
        Get the best model path for an imaging paradigm.

        Returns the model_path from the paradigm-best classification run.

        Args:
            imaging_paradigm: e.g., "1p625x_z4"

        Returns:
            Model path string, or None
        """
        best = self.get_best_for_paradigm(imaging_paradigm, exp_type="classification")
        if not best:
            return None
        return best.get('class_model_path') or best.get('model_path')

    def update_rating(self, exp_id: str, rating: int, notes: Optional[str] = None) -> bool:
        """
        Update experiment rating (1-5) with optional notes.

        This is a convenience alias for rate_experiment.
        """
        return self.rate_experiment(exp_id, rating, notes)

    def log_comparison(
        self,
        exp_id: str,
        ref_id: str,
        gained: int,
        lost: int,
    ) -> bool:
        """
        Log comparison results between two experiments.

        Args:
            exp_id: The experiment being evaluated
            ref_id: The reference experiment (e.g., current best)
            gained: Number of points gained vs reference
            lost: Number of points lost vs reference

        Returns:
            True if successful
        """
        return self._update_row(exp_id, {
            'comparison_ref': ref_id,
            'diff_gained': gained,
            'diff_lost': lost,
            'diff_vs_best': gained + lost,  # Total difference
        })

    def get_best_for_brain(self, brain: str, exp_type: str = 'detection') -> Optional[Dict[str, str]]:
        """
        Get the marked-best experiment for a specific brain.

        Args:
            brain: Brain name to search for
            exp_type: Experiment type ('detection', 'classification', etc.)

        Returns:
            The best experiment dict, or None if not found
        """
        rows = self._read_all()
        for row in rows:
            if (row.get('brain') == brain and
                row.get('exp_type') == exp_type and
                row.get('marked_best') == 'True'):
                return row

        # Fall back to highest-rated if no marked best
        best_rows = self.get_best(exp_type=exp_type, brain=brain, limit=1)
        return best_rows[0] if best_rows else None


# =============================================================================
# CLI INTERFACE
# =============================================================================

def print_experiment_row(row: Dict[str, str], verbose: bool = False):
    """Print a single experiment row in a readable format."""
    exp_id = row.get('exp_id', '?')
    exp_type = row.get('exp_type', '?')
    brain = row.get('brain', '?')
    status = row.get('status', '?')
    created = row.get('created_at', '?')[:16]  # Trim to date + time
    rating = row.get('rating', '')

    # Status symbols (ASCII-safe for Windows)
    status_symbols = {
        'completed': '[OK]',
        'failed': '[X]',
        'started': '[..]',
        'running': '[>>]',
        'cancelled': '[--]',
        'approved': '[OK]',
    }
    symbol = status_symbols.get(status, '[?]')

    # Rating stars (ASCII-safe)
    if rating and rating.isdigit():
        r = int(rating)
        stars = '*' * r + '.' * (5 - r)
    else:
        stars = '-----'

    # Type abbreviation
    type_abbrev = {'detection': 'DET', 'training': 'TRN', 'classification': 'CLS', 'registration': 'REG', 'counts': 'CNT'}
    type_str = type_abbrev.get(exp_type, exp_type[:3].upper())

    print(f"  {symbol} [{type_str}] {exp_id}  {brain}  {stars}  {created}")

    if verbose:
        # Show more details
        duration = row.get('duration_seconds', '')
        if duration:
            mins = float(duration) / 60
            print(f"           Duration: {mins:.1f} min")

        cells = row.get('det_cells_found') or row.get('class_cells_found') or row.get('count_total_cells')
        if cells:
            print(f"           Cells: {cells}")

        notes = row.get('notes', '')
        if notes:
            print(f"           Notes: {notes[:60]}...")


def main():
    """CLI interface for calibration run tracker."""
    import argparse

    parser = argparse.ArgumentParser(
        description='BrainGlobe Calibration Run Tracker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tracks CALIBRATION runs (tuning/optimization experiments).
For PRODUCTION results, see region_counts.csv.

Examples:
    python -m mousebrain.tracker                    # Show recent runs
    python -m mousebrain.tracker --search "349"     # Search by brain
    python -m mousebrain.tracker --type detection   # Filter by type
    python -m mousebrain.tracker --stats            # Show statistics
    python -m mousebrain.tracker --best detection   # Best detection runs
        """
    )

    parser.add_argument('--search', '-s', help='Search term (brain name, exp_id, etc.)')
    parser.add_argument('--type', '-t', choices=EXP_TYPES, help='Filter by experiment type')
    parser.add_argument('--status', choices=['completed', 'failed', 'started', 'running'])
    parser.add_argument('--best', metavar='TYPE', help='Show best-rated of a type')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--limit', '-n', type=int, default=20, help='Number of results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show more details')
    parser.add_argument('--csv', help='Path to experiments CSV (default: standard location)')

    args = parser.parse_args()

    # Initialize tracker
    tracker = ExperimentTracker(args.csv)

    print("=" * 70)
    print("BrainGlobe Calibration Run Tracker")
    print(f"CSV: {tracker.csv_path}")
    print("(For production results, see region_counts.csv)")
    print("=" * 70)

    if args.stats:
        stats = tracker.get_statistics()
        print(f"\nTotal calibration runs: {stats['total']}")
        print(f"Rated: {stats['rated']} (avg: {stats['avg_rating']:.1f}★)")

        print("\nBy type:")
        for t, count in sorted(stats['by_type'].items()):
            print(f"  {t}: {count}")

        print("\nBy status:")
        for s, count in sorted(stats['by_status'].items()):
            print(f"  {s}: {count}")

        print("\nBy brain (top 10):")
        sorted_brains = sorted(stats['by_brain'].items(), key=lambda x: x[1], reverse=True)[:10]
        for b, count in sorted_brains:
            print(f"  {b}: {count}")
        return

    if args.best:
        print(f"\nBest {args.best} runs:")
        rows = tracker.get_best(args.best, limit=args.limit)
    elif args.search or args.type or args.status:
        print(f"\nSearch results:")
        rows = tracker.search(
            brain=args.search,
            exp_type=args.type,
            status=args.status,
            limit=args.limit
        )
    else:
        print(f"\nRecent calibration runs:")
        rows = tracker.get_recent(limit=args.limit)

    if not rows:
        print("  No calibration runs found.")
    else:
        for row in rows:
            print_experiment_row(row, verbose=args.verbose)

    print()


if __name__ == '__main__':
    main()
