#!/usr/bin/env python3
"""
experiment_tracker.py

Central experiment tracking for BrainGlobe pipeline optimization.

Tracks all runs of:
- brainreg (atlas registration)
- cellfinder detection (candidate cell finding)
- cellfinder training (classification network training)
- cellfinder classification (filtering candidates)
- brainglobe-segmentation (regional cell counts)
- crop optimization (finding optimal crop position)

Stores everything in a human-readable CSV that you can open in Excel,
filter, sort, and annotate.

Usage:
    from experiment_tracker import ExperimentTracker
    
    tracker = ExperimentTracker()
    
    # Log a detection run
    exp_id = tracker.log_detection(
        brain="349_CNT_01_02_1p625x_z4",
        signal_channel="ch0",
        background_channel="ch1",
        soma_diameter=16,
        ball_xy_size=6,
        ball_z_size=15,
        output_path="/path/to/output",
        duration_seconds=3600,
        candidates_found=15000,
        notes="First attempt with default settings"
    )
    
    # Later, add a rating
    tracker.rate_experiment(exp_id, rating=3, notes="Too many false positives")
    
    # Query experiments
    detections = tracker.get_experiments(experiment_type="detection", brain="349*")
    
    # Log a crop optimization run
    exp_id = tracker.log_crop(
        brain="349_CNT_01_02_1p625x_z4",
        start_y=5000,
        optimal_y=6234,
        quality_score=0.87,
        combined_score=0.82,
        iterations=8,
        algorithm="hill_climbing",
        crop_penalty=0.1,
        notes="Converged after 8 iterations"
    )
"""

import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
import json
import hashlib

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default location for experiment log (in dev scripts folder)
DEFAULT_LOG_PATH = Path(r"Y:\2_Connectome\3_Nuclei_Detection\util_Brainglobe\util_DevScripts\experiment_log.csv")

# Experiment types
EXPERIMENT_TYPES = [
    "registration",
    "detection", 
    "training",
    "classification",
    "counts",
    "crop",
]

# =============================================================================
# CSV COLUMNS
# =============================================================================

# Core columns (all experiment types)
CORE_COLUMNS = [
    "experiment_id",      # Unique ID (timestamp + hash)
    "experiment_type",    # registration, detection, training, classification, counts, crop
    "timestamp",          # ISO format datetime
    "brain",              # Brain/pipeline identifier
    "status",             # started, completed, failed
    "duration_seconds",   # How long it took
    "output_path",        # Where results are stored
    "rating",             # User rating (1-5, empty = unrated)
    "notes",              # Free text notes
    "parent_experiment",  # Links to prior experiment (e.g., classification → detection)
]

# Registration-specific columns
REGISTRATION_COLUMNS = [
    "reg_atlas",              # e.g., allen_mouse_25um
    "reg_orientation",        # e.g., iar
    "reg_voxel_z",
    "reg_voxel_y", 
    "reg_voxel_x",
    "reg_additional_args",    # Any extra brainreg arguments
]

# Detection-specific columns
DETECTION_COLUMNS = [
    "det_signal_channel",
    "det_background_channel",
    "det_soma_diameter",
    "det_ball_xy_size",
    "det_ball_z_size",
    "det_ball_overlap_fraction",
    "det_max_cluster_size",
    "det_soma_spread_factor",
    "det_log_sigma_size",
    "det_n_free_cpus",
    "det_additional_args",
    "det_candidates_found",   # Result: how many candidates
]

# Training-specific columns
TRAINING_COLUMNS = [
    "train_network_depth",    # ResNet depth (18, 34, 50, etc.)
    "train_learning_rate",
    "train_batch_size",
    "train_epochs",
    "train_continue_from",    # Path to existing model if continuing
    "train_num_positive",     # Number of positive examples
    "train_num_negative",     # Number of negative examples
    "train_augment",          # Augmentation settings
    "train_additional_args",
    "train_final_loss",       # Result: final training loss
    "train_final_accuracy",   # Result: final accuracy
]

# Classification-specific columns
CLASSIFICATION_COLUMNS = [
    "class_model_path",       # Which trained model
    "class_candidates_path",  # Input candidates
    "class_cube_size",
    "class_batch_size",
    "class_n_free_cpus",
    "class_additional_args",
    "class_cells_found",      # Result: cells after classification
    "class_rejected",         # Result: rejected candidates
]

# Counts-specific columns
COUNTS_COLUMNS = [
    "counts_registration",    # Which registration to use
    "counts_cells_path",      # Which classified cells
    "counts_atlas",
    "counts_additional_args",
    "counts_total_cells",     # Result: total cell count
    "counts_regions_file",    # Result: path to regional counts CSV
]

# Crop optimization-specific columns
CROP_COLUMNS = [
    "crop_start_y",           # Initial Y position tested
    "crop_optimal_y",         # Result: optimal Y position found
    "crop_start_pct",         # Initial percentage from top
    "crop_optimal_pct",       # Result: optimal percentage from top
    "crop_quality_score",     # Result: registration quality score (0-1)
    "crop_combined_score",    # Result: quality minus crop penalty
    "crop_penalty_weight",    # Penalty weight for tissue removal
    "crop_iterations",        # Number of iterations to converge
    "crop_algorithm",         # hill_climbing, grid_search, manual
    "crop_atlas",             # Atlas used for quality assessment
    "crop_regions_evaluated", # Which brainstem regions were checked
    "crop_region_scores",     # JSON of individual region scores
    "crop_step_history",      # JSON of optimization path
]

# All columns combined
ALL_COLUMNS = (CORE_COLUMNS + REGISTRATION_COLUMNS + DETECTION_COLUMNS + 
               TRAINING_COLUMNS + CLASSIFICATION_COLUMNS + COUNTS_COLUMNS +
               CROP_COLUMNS)


# =============================================================================
# EXPERIMENT TRACKER CLASS
# =============================================================================

class ExperimentTracker:
    """
    Track and query BrainGlobe pipeline experiments.
    
    All data stored in a single CSV file for easy viewing in Excel.
    """
    
    def __init__(self, log_path: Optional[Path] = None):
        """
        Initialize tracker.
        
        Args:
            log_path: Path to CSV log file. Default: experiment_log.csv in root folder
        """
        self.log_path = Path(log_path) if log_path else DEFAULT_LOG_PATH
        self._ensure_log_exists()
    
    def _ensure_log_exists(self):
        """Create log file with headers if it doesn't exist."""
        if not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
                writer.writeheader()
            print(f"Created experiment log: {self.log_path}")
        else:
            # Check if we need to add new columns (for upgrades)
            self._migrate_columns_if_needed()
    
    def _migrate_columns_if_needed(self):
        """Add any new columns to existing CSV (for version upgrades)."""
        try:
            with open(self.log_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_columns = reader.fieldnames or []
            
            # Check for missing columns
            missing = [col for col in ALL_COLUMNS if col not in existing_columns]
            
            if missing:
                print(f"Adding {len(missing)} new columns to experiment log...")
                rows = self._read_all()
                
                # Rewrite with all columns
                with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
                    writer.writeheader()
                    for row in rows:
                        full_row = {col: row.get(col, "") for col in ALL_COLUMNS}
                        writer.writerow(full_row)
                
                print(f"  Added columns: {', '.join(missing)}")
        except Exception as e:
            print(f"Warning: Could not check/migrate columns: {e}")
    
    def _generate_id(self, experiment_type: str) -> str:
        """Generate a unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Add a short hash for uniqueness even within same second
        hash_input = f"{timestamp}_{experiment_type}_{os.getpid()}"
        short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:6]
        return f"{experiment_type[:3]}_{timestamp}_{short_hash}"
    
    def _write_row(self, row: Dict[str, Any]):
        """Append a row to the CSV."""
        with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
            # Fill missing columns with empty string
            full_row = {col: row.get(col, "") for col in ALL_COLUMNS}
            writer.writerow(full_row)
    
    def _read_all(self) -> List[Dict[str, str]]:
        """Read all rows from CSV."""
        if not self.log_path.exists():
            return []
        with open(self.log_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    
    def _update_row(self, experiment_id: str, updates: Dict[str, Any]):
        """Update an existing row by experiment_id."""
        rows = self._read_all()
        updated = False
        
        for row in rows:
            if row['experiment_id'] == experiment_id:
                row.update({k: str(v) if v is not None else "" for k, v in updates.items()})
                updated = True
                break
        
        if updated:
            with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
        
        return updated
    
    # =========================================================================
    # LOGGING METHODS
    # =========================================================================
    
    def log_registration(
        self,
        brain: str,
        atlas: str = "allen_mouse_25um",
        orientation: str = "iar",
        voxel_z: float = None,
        voxel_y: float = None,
        voxel_x: float = None,
        output_path: str = None,
        additional_args: str = None,
        status: str = "started",
        duration_seconds: float = None,
        notes: str = None,
    ) -> str:
        """
        Log a brainreg registration run.
        
        Returns experiment_id for later updates.
        """
        exp_id = self._generate_id("registration")
        
        row = {
            'experiment_id': exp_id,
            'experiment_type': 'registration',
            'timestamp': datetime.now().isoformat(),
            'brain': brain,
            'status': status,
            'duration_seconds': duration_seconds or "",
            'output_path': output_path or "",
            'notes': notes or "",
            'reg_atlas': atlas,
            'reg_orientation': orientation,
            'reg_voxel_z': voxel_z or "",
            'reg_voxel_y': voxel_y or "",
            'reg_voxel_x': voxel_x or "",
            'reg_additional_args': additional_args or "",
        }
        
        self._write_row(row)
        print(f"Logged registration: {exp_id}")
        return exp_id
    
    def log_detection(
        self,
        brain: str,
        signal_channel: str = None,
        background_channel: str = None,
        soma_diameter: int = None,
        ball_xy_size: int = None,
        ball_z_size: int = None,
        ball_overlap_fraction: float = None,
        max_cluster_size: int = None,
        soma_spread_factor: float = None,
        log_sigma_size: float = None,
        n_free_cpus: int = None,
        output_path: str = None,
        additional_args: str = None,
        status: str = "started",
        duration_seconds: float = None,
        candidates_found: int = None,
        parent_experiment: str = None,
        notes: str = None,
    ) -> str:
        """
        Log a cellfinder detection run.
        
        Returns experiment_id for later updates.
        """
        exp_id = self._generate_id("detection")
        
        row = {
            'experiment_id': exp_id,
            'experiment_type': 'detection',
            'timestamp': datetime.now().isoformat(),
            'brain': brain,
            'status': status,
            'duration_seconds': duration_seconds or "",
            'output_path': output_path or "",
            'parent_experiment': parent_experiment or "",
            'notes': notes or "",
            'det_signal_channel': signal_channel or "",
            'det_background_channel': background_channel or "",
            'det_soma_diameter': soma_diameter or "",
            'det_ball_xy_size': ball_xy_size or "",
            'det_ball_z_size': ball_z_size or "",
            'det_ball_overlap_fraction': ball_overlap_fraction or "",
            'det_max_cluster_size': max_cluster_size or "",
            'det_soma_spread_factor': soma_spread_factor or "",
            'det_log_sigma_size': log_sigma_size or "",
            'det_n_free_cpus': n_free_cpus or "",
            'det_additional_args': additional_args or "",
            'det_candidates_found': candidates_found or "",
        }
        
        self._write_row(row)
        print(f"Logged detection: {exp_id}")
        return exp_id
    
    def log_training(
        self,
        brain: str,
        network_depth: int = None,
        learning_rate: float = None,
        batch_size: int = None,
        epochs: int = None,
        continue_from: str = None,
        num_positive: int = None,
        num_negative: int = None,
        augment: str = None,
        output_path: str = None,
        additional_args: str = None,
        status: str = "started",
        duration_seconds: float = None,
        final_loss: float = None,
        final_accuracy: float = None,
        notes: str = None,
    ) -> str:
        """
        Log a cellfinder training run.
        
        Returns experiment_id for later updates.
        """
        exp_id = self._generate_id("training")
        
        row = {
            'experiment_id': exp_id,
            'experiment_type': 'training',
            'timestamp': datetime.now().isoformat(),
            'brain': brain,
            'status': status,
            'duration_seconds': duration_seconds or "",
            'output_path': output_path or "",
            'notes': notes or "",
            'train_network_depth': network_depth or "",
            'train_learning_rate': learning_rate or "",
            'train_batch_size': batch_size or "",
            'train_epochs': epochs or "",
            'train_continue_from': continue_from or "",
            'train_num_positive': num_positive or "",
            'train_num_negative': num_negative or "",
            'train_augment': augment or "",
            'train_additional_args': additional_args or "",
            'train_final_loss': final_loss or "",
            'train_final_accuracy': final_accuracy or "",
        }
        
        self._write_row(row)
        print(f"Logged training: {exp_id}")
        return exp_id
    
    def log_classification(
        self,
        brain: str,
        model_path: str = None,
        candidates_path: str = None,
        cube_size: int = None,
        batch_size: int = None,
        n_free_cpus: int = None,
        output_path: str = None,
        additional_args: str = None,
        status: str = "started",
        duration_seconds: float = None,
        cells_found: int = None,
        rejected: int = None,
        parent_experiment: str = None,
        notes: str = None,
    ) -> str:
        """
        Log a cellfinder classification run.
        
        Returns experiment_id for later updates.
        """
        exp_id = self._generate_id("classification")
        
        row = {
            'experiment_id': exp_id,
            'experiment_type': 'classification',
            'timestamp': datetime.now().isoformat(),
            'brain': brain,
            'status': status,
            'duration_seconds': duration_seconds or "",
            'output_path': output_path or "",
            'parent_experiment': parent_experiment or "",
            'notes': notes or "",
            'class_model_path': model_path or "",
            'class_candidates_path': candidates_path or "",
            'class_cube_size': cube_size or "",
            'class_batch_size': batch_size or "",
            'class_n_free_cpus': n_free_cpus or "",
            'class_additional_args': additional_args or "",
            'class_cells_found': cells_found or "",
            'class_rejected': rejected or "",
        }
        
        self._write_row(row)
        print(f"Logged classification: {exp_id}")
        return exp_id
    
    def log_counts(
        self,
        brain: str,
        registration_path: str = None,
        cells_path: str = None,
        atlas: str = None,
        output_path: str = None,
        additional_args: str = None,
        status: str = "started",
        duration_seconds: float = None,
        total_cells: int = None,
        regions_file: str = None,
        parent_experiment: str = None,
        notes: str = None,
    ) -> str:
        """
        Log a regional cell counts run.
        
        Returns experiment_id for later updates.
        """
        exp_id = self._generate_id("counts")
        
        row = {
            'experiment_id': exp_id,
            'experiment_type': 'counts',
            'timestamp': datetime.now().isoformat(),
            'brain': brain,
            'status': status,
            'duration_seconds': duration_seconds or "",
            'output_path': output_path or "",
            'parent_experiment': parent_experiment or "",
            'notes': notes or "",
            'counts_registration': registration_path or "",
            'counts_cells_path': cells_path or "",
            'counts_atlas': atlas or "",
            'counts_additional_args': additional_args or "",
            'counts_total_cells': total_cells or "",
            'counts_regions_file': regions_file or "",
        }
        
        self._write_row(row)
        print(f"Logged counts: {exp_id}")
        return exp_id
    
    def log_crop(
        self,
        brain: str,
        start_y: int = None,
        optimal_y: int = None,
        start_pct: float = None,
        optimal_pct: float = None,
        quality_score: float = None,
        combined_score: float = None,
        penalty_weight: float = None,
        iterations: int = None,
        algorithm: str = "hill_climbing",
        atlas: str = "allen_mouse_50um",
        regions_evaluated: str = None,
        region_scores: dict = None,
        step_history: list = None,
        output_path: str = None,
        status: str = "started",
        duration_seconds: float = None,
        notes: str = None,
    ) -> str:
        """
        Log a crop optimization run.
        
        This tracks the iterative process of finding the optimal Y-crop
        position for a brain sample, balancing registration quality
        against preserving brain tissue.
        
        Args:
            brain: Brain/pipeline identifier
            start_y: Initial Y position tested
            optimal_y: Final optimal Y position found
            start_pct: Initial percentage from top (0-100)
            optimal_pct: Final optimal percentage from top
            quality_score: Registration quality score (0-1)
            combined_score: Quality minus crop penalty
            penalty_weight: Penalty weight used (typically 0.1)
            iterations: Number of iterations to converge
            algorithm: Algorithm used (hill_climbing, grid_search, manual)
            atlas: Atlas used for quality assessment
            regions_evaluated: Comma-separated list of regions checked
            region_scores: Dict of region_name -> score
            step_history: List of (y, score) tuples showing optimization path
            output_path: Path to cropped output
            status: started, completed, failed
            duration_seconds: How long it took
            notes: Free text notes
        
        Returns:
            experiment_id for later updates
        """
        exp_id = self._generate_id("crop")
        
        row = {
            'experiment_id': exp_id,
            'experiment_type': 'crop',
            'timestamp': datetime.now().isoformat(),
            'brain': brain,
            'status': status,
            'duration_seconds': duration_seconds or "",
            'output_path': output_path or "",
            'notes': notes or "",
            'crop_start_y': start_y or "",
            'crop_optimal_y': optimal_y or "",
            'crop_start_pct': start_pct or "",
            'crop_optimal_pct': optimal_pct or "",
            'crop_quality_score': quality_score or "",
            'crop_combined_score': combined_score or "",
            'crop_penalty_weight': penalty_weight or "",
            'crop_iterations': iterations or "",
            'crop_algorithm': algorithm or "",
            'crop_atlas': atlas or "",
            'crop_regions_evaluated': regions_evaluated or "",
            'crop_region_scores': json.dumps(region_scores) if region_scores else "",
            'crop_step_history': json.dumps(step_history) if step_history else "",
        }
        
        self._write_row(row)
        print(f"Logged crop optimization: {exp_id}")
        return exp_id
    
    # =========================================================================
    # UPDATE METHODS
    # =========================================================================
    
    def update_status(
        self, 
        experiment_id: str, 
        status: str,
        duration_seconds: float = None,
        **result_fields
    ):
        """
        Update experiment status and results.
        
        Args:
            experiment_id: The experiment to update
            status: New status (completed, failed, etc.)
            duration_seconds: How long it took
            **result_fields: Any result fields to update (e.g., candidates_found=15000)
        """
        updates = {'status': status}
        if duration_seconds is not None:
            updates['duration_seconds'] = duration_seconds
        updates.update(result_fields)
        
        if self._update_row(experiment_id, updates):
            print(f"Updated {experiment_id}: {status}")
        else:
            print(f"Warning: Experiment {experiment_id} not found")
    
    def rate_experiment(
        self,
        experiment_id: str,
        rating: int,
        notes: str = None
    ):
        """
        Add rating and notes to an experiment.
        
        Args:
            experiment_id: The experiment to rate
            rating: 1-5 scale (1=bad, 5=excellent)
            notes: Optional notes to add/append
        """
        if not 1 <= rating <= 5:
            print("Warning: Rating should be 1-5")
        
        updates = {'rating': rating}
        
        if notes:
            # Get existing notes and append
            rows = self._read_all()
            for row in rows:
                if row['experiment_id'] == experiment_id:
                    existing = row.get('notes', '')
                    if existing:
                        updates['notes'] = f"{existing} | {notes}"
                    else:
                        updates['notes'] = notes
                    break
        
        if self._update_row(experiment_id, updates):
            print(f"Rated {experiment_id}: {rating}/5")
        else:
            print(f"Warning: Experiment {experiment_id} not found")
    
    def add_notes(self, experiment_id: str, notes: str):
        """Add notes to an experiment (appends to existing)."""
        rows = self._read_all()
        for row in rows:
            if row['experiment_id'] == experiment_id:
                existing = row.get('notes', '')
                if existing:
                    new_notes = f"{existing} | {notes}"
                else:
                    new_notes = notes
                self._update_row(experiment_id, {'notes': new_notes})
                print(f"Added notes to {experiment_id}")
                return
        print(f"Warning: Experiment {experiment_id} not found")
    
    # =========================================================================
    # QUERY METHODS
    # =========================================================================
    
    def get_experiments(
        self,
        experiment_type: str = None,
        brain: str = None,
        status: str = None,
        rated_only: bool = False,
        unrated_only: bool = False,
        min_rating: int = None,
        limit: int = None,
    ) -> List[Dict[str, str]]:
        """
        Query experiments with optional filters.
        
        Args:
            experiment_type: Filter by type (registration, detection, etc.)
            brain: Filter by brain (supports * wildcard)
            status: Filter by status
            rated_only: Only return rated experiments
            unrated_only: Only return unrated experiments
            min_rating: Only return experiments with rating >= this
            limit: Maximum number to return (most recent first)
        
        Returns:
            List of experiment dicts matching filters
        """
        rows = self._read_all()
        
        # Apply filters
        filtered = []
        for row in rows:
            if experiment_type and row.get('experiment_type') != experiment_type:
                continue
            if brain:
                if '*' in brain:
                    pattern = brain.replace('*', '')
                    if pattern not in row.get('brain', ''):
                        continue
                elif row.get('brain') != brain:
                    continue
            if status and row.get('status') != status:
                continue
            if rated_only and not row.get('rating'):
                continue
            if unrated_only and row.get('rating'):
                continue
            if min_rating:
                try:
                    if int(row.get('rating', 0)) < min_rating:
                        continue
                except ValueError:
                    continue
            filtered.append(row)
        
        # Sort by timestamp descending (most recent first)
        filtered.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Apply limit
        if limit:
            filtered = filtered[:limit]
        
        return filtered
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, str]]:
        """Get a single experiment by ID."""
        rows = self._read_all()
        for row in rows:
            if row['experiment_id'] == experiment_id:
                return row
        return None
    
    def get_best(
        self,
        experiment_type: str,
        n: int = 5,
        status: str = "completed"
    ) -> List[Dict[str, str]]:
        """
        Get top N rated experiments of a given type.
        
        Args:
            experiment_type: Type to filter by
            n: Number to return
            status: Status filter (default: completed)
        
        Returns:
            List of experiments sorted by rating (highest first)
        """
        experiments = self.get_experiments(
            experiment_type=experiment_type,
            status=status,
            rated_only=True,
        )
        
        # Sort by rating descending
        experiments.sort(key=lambda x: int(x.get('rating', 0)), reverse=True)
        
        return experiments[:n]
    
    def get_crop_history(self, brain: str) -> List[Dict[str, str]]:
        """
        Get all crop optimization runs for a brain.
        
        Useful for seeing the history of crop attempts and finding
        what worked best.
        """
        return self.get_experiments(
            experiment_type="crop",
            brain=brain,
        )
    
    def get_best_crop(self, brain: str) -> Optional[Dict[str, str]]:
        """
        Get the best crop optimization result for a brain.
        
        Returns the completed crop run with the highest combined score.
        """
        crops = self.get_crop_history(brain)
        
        completed = [c for c in crops if c.get('status') == 'completed']
        if not completed:
            return None
        
        # Sort by combined score descending
        completed.sort(
            key=lambda x: float(x.get('crop_combined_score', 0) or 0),
            reverse=True
        )
        
        return completed[0] if completed else None
    
    # =========================================================================
    # DISPLAY METHODS
    # =========================================================================
    
    def print_experiment(self, experiment_id: str):
        """Print detailed view of a single experiment."""
        exp = self.get_experiment(experiment_id)
        if not exp:
            print(f"Experiment not found: {experiment_id}")
            return
        
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_id}")
        print(f"{'='*60}")
        
        # Determine which columns are relevant
        exp_type = exp.get('experiment_type', '')
        
        # Always show core info
        print(f"\nType: {exp_type}")
        print(f"Brain: {exp.get('brain', '-')}")
        print(f"Status: {exp.get('status', '-')}")
        print(f"Timestamp: {exp.get('timestamp', '-')}")
        
        duration = exp.get('duration_seconds')
        if duration:
            try:
                mins = float(duration) / 60
                print(f"Duration: {mins:.1f} minutes")
            except:
                print(f"Duration: {duration}")
        
        print(f"Rating: {exp.get('rating') + '/5' if exp.get('rating') else 'Unrated'}")
        
        if exp.get('notes'):
            print(f"Notes: {exp.get('notes')}")
        
        if exp.get('output_path'):
            print(f"Output: {exp.get('output_path')}")
        
        if exp.get('parent_experiment'):
            print(f"Parent: {exp.get('parent_experiment')}")
        
        # Type-specific details
        print(f"\n--- {exp_type.title()} Details ---")
        
        if exp_type == 'detection':
            fields = ['det_signal_channel', 'det_background_channel', 'det_soma_diameter',
                     'det_ball_xy_size', 'det_ball_z_size', 'det_ball_overlap_fraction',
                     'det_max_cluster_size', 'det_candidates_found']
        elif exp_type == 'training':
            fields = ['train_network_depth', 'train_learning_rate', 'train_batch_size',
                     'train_epochs', 'train_num_positive', 'train_num_negative',
                     'train_final_loss', 'train_final_accuracy']
        elif exp_type == 'classification':
            fields = ['class_model_path', 'class_cube_size', 'class_batch_size',
                     'class_cells_found', 'class_rejected']
        elif exp_type == 'registration':
            fields = ['reg_atlas', 'reg_orientation', 'reg_voxel_z', 'reg_voxel_y',
                     'reg_voxel_x']
        elif exp_type == 'counts':
            fields = ['counts_atlas', 'counts_total_cells', 'counts_regions_file']
        elif exp_type == 'crop':
            fields = ['crop_start_y', 'crop_optimal_y', 'crop_start_pct', 'crop_optimal_pct',
                     'crop_quality_score', 'crop_combined_score', 'crop_penalty_weight',
                     'crop_iterations', 'crop_algorithm', 'crop_atlas', 'crop_regions_evaluated']
        else:
            fields = []
        
        for field in fields:
            val = exp.get(field)
            if val:
                # Clean up field name for display
                display_name = field.split('_', 1)[1] if '_' in field else field
                display_name = display_name.replace('_', ' ').title()
                print(f"  {display_name}: {val}")
        
        print()


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Simple CLI for quick queries."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Experiment tracker CLI')
    parser.add_argument('--list', '-l', action='store_true', help='List recent experiments')
    parser.add_argument('--type', '-t', choices=EXPERIMENT_TYPES, help='Filter by type')
    parser.add_argument('--brain', '-b', help='Filter by brain')
    parser.add_argument('--show', '-s', help='Show experiment details')
    parser.add_argument('--log', help='Path to log file')
    
    args = parser.parse_args()
    
    tracker = ExperimentTracker(log_path=args.log)
    
    if args.show:
        tracker.print_experiment(args.show)
    elif args.list or args.type or args.brain:
        experiments = tracker.get_experiments(
            experiment_type=args.type,
            brain=args.brain,
            limit=20,
        )
        
        if experiments:
            print(f"\n{'ID':<30} {'Type':<15} {'Brain':<25} {'Status':<10}")
            print("-" * 80)
            for exp in experiments:
                print(f"{exp['experiment_id']:<30} {exp.get('experiment_type', '-'):<15} "
                      f"{exp.get('brain', '-'):<25} {exp.get('status', '-'):<10}")
        else:
            print("No experiments found.")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
