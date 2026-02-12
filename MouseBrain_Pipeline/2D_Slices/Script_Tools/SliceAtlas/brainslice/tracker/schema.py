"""
schema.py - CSV column definitions for BrainSlice calibration run tracking.

Defines the structure for tracking:
- Registration runs (slice-to-atlas alignment)
- Detection runs (nuclei detection)
- Colocalization runs (intensity measurement + classification)
- Quantification runs (regional counting)
"""

# Run types
RUN_TYPES = ["registration", "detection", "colocalization", "quantification"]

# CSV columns (order matters for readability)
CSV_COLUMNS = [
    # Identity
    "run_id",
    "run_type",  # registration, detection, colocalization, quantification
    "sample_id",
    "created_at",

    # Sample hierarchy (auto-filled from sample name)
    "project",
    "cohort",
    "slice_num",

    # Status
    "status",  # started, running, completed, failed, cancelled
    "duration_seconds",

    # Image metadata
    "image_path",
    "image_channels",  # JSON list of channel names
    "pixel_size_um",  # Pixel size in microns

    # Registration parameters
    "reg_atlas",  # allen_mouse_10um, allen_spinal_cord_20um, etc.
    "reg_orientation",  # coronal, sagittal, horizontal
    "reg_method",  # abba_manual, abba_auto, deepslice
    "reg_ap_position",  # Anterior-posterior position in atlas (um)
    "reg_transform_path",  # Path to saved transform
    "reg_quality_score",  # Manual QC rating 1-5

    # Detection parameters
    "det_channel",  # Which channel was used (e.g., "red", 0)
    "det_model",  # StarDist model name
    "det_prob_thresh",
    "det_nms_thresh",
    "det_scale",
    "det_min_area",
    "det_max_area",
    "det_nuclei_found",
    "det_backend",  # 'stardist' or 'cellpose'
    "det_preprocessing",  # JSON string of preprocessing params used
    "det_raw_count",  # count before any filtering
    "det_removed_by_size",  # count removed by size filter
    "det_removed_by_border",  # count removed by border filter
    "det_removed_by_morphology",  # count removed by morphology filter

    # Colocalization parameters
    "coloc_signal_channel",  # Which channel to measure (e.g., "green", 1)
    "coloc_background_method",  # percentile, mode, tissue_mask
    "coloc_background_percentile",
    "coloc_background_value",
    "coloc_threshold_method",  # fold_change, std_above, absolute
    "coloc_threshold_value",
    "coloc_positive_cells",
    "coloc_negative_cells",
    "coloc_positive_fraction",

    # Quantification
    "quant_total_regions",
    "quant_top_region",  # Region with most cells
    "quant_top_region_count",

    # Paths
    "input_path",
    "output_path",
    "labels_path",  # Path to nuclei labels image
    "measurements_path",  # Path to per-cell measurements CSV
    "region_counts_path",  # Path to per-region counts CSV

    # Linkage
    "parent_run",  # Links colocalization to detection, etc.

    # User feedback
    "marked_best",  # True if user marked this as best for sample
    "rating",  # 1-5 stars
    "notes",
    "tags",

    # Metadata
    "script_version",
    "hostname",
]

# Column groups for different run types
REGISTRATION_COLUMNS = [
    "reg_atlas", "reg_orientation", "reg_method", "reg_ap_position",
    "reg_transform_path", "reg_quality_score"
]

DETECTION_COLUMNS = [
    "det_channel", "det_model", "det_prob_thresh", "det_nms_thresh",
    "det_scale", "det_min_area", "det_max_area", "det_nuclei_found",
    "det_backend", "det_preprocessing", "det_raw_count",
    "det_removed_by_size", "det_removed_by_border", "det_removed_by_morphology"
]

COLOCALIZATION_COLUMNS = [
    "coloc_signal_channel", "coloc_background_method", "coloc_background_percentile",
    "coloc_background_value", "coloc_threshold_method", "coloc_threshold_value",
    "coloc_positive_cells", "coloc_negative_cells", "coloc_positive_fraction"
]

QUANTIFICATION_COLUMNS = [
    "quant_total_regions", "quant_top_region", "quant_top_region_count"
]
