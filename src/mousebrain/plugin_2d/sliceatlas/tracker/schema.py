"""
schema.py - CSV column definitions for BrainSlice calibration run tracking.

Defines the structure for tracking:
- Registration runs (slice-to-atlas alignment)
- Detection runs (nuclei detection)
- Colocalization runs (intensity measurement + classification)
- Quantification runs (regional counting)
"""

# Run types
RUN_TYPES = ["registration", "detection", "colocalization", "quantification", "particle_analysis"]

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

    # Particle analysis parameters
    "pa_detection_channel",   # Which channel was binarized
    "pa_threshold_value",     # Hard threshold used for binarization
    "pa_measurement_channel", # Which channel intensities were measured in
    "pa_min_area",
    "pa_max_area",
    "pa_min_circularity",
    "pa_max_circularity",
    "pa_particles_found",     # Count after filtering
    "pa_particles_raw",       # Count before filtering (from binary mask)
    "pa_bg_method",           # Background estimation method
    "pa_bg_value",            # Estimated background value
    "pa_mean_intensity",      # Mean of all particle mean intensities

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

PARTICLE_ANALYSIS_COLUMNS = [
    "pa_detection_channel", "pa_threshold_value", "pa_measurement_channel",
    "pa_min_area", "pa_max_area", "pa_min_circularity", "pa_max_circularity",
    "pa_particles_found", "pa_particles_raw", "pa_bg_method", "pa_bg_value",
    "pa_mean_intensity"
]

# script_version written on colocalization rows whose coloc_threshold_value is
# the value classification actually used. For the background_mean method that is
# the SIGMA threshold (positive if signal > background mean + sigma * std).
# WHY a version marker: rows written before it by the napari widget stored the
# fold-change spin box in coloc_threshold_value even when background_mean was
# chosen, so the number in an old row cannot be trusted as a sigma. The batch
# (batch_encr.get_tuned_parameters) only takes a threshold from a row carrying
# this version; an older row is refused with a message saying to re-run it.
COLOC_SIGMA_RECORDED_VERSION = "1.1.0"


def coloc_fold_change_threshold(row) -> "float | None":
    """The fold-change threshold a colocalization row records, or None.

    WHY: from COLOC_SIGMA_RECORDED_VERSION on, a background_mean row's
    coloc_threshold_value is a SIGMA (standard deviations above the background
    mean), not a fold change. Code that draws a fold-change threshold (the
    widget's diagnostic plots after "Load selected run") must not draw a sigma
    as if it were one; it gets None and falls back to its own fold-change
    setting, exactly as it does right after a live run.
    """
    if (row.get('coloc_threshold_method') == 'background_mean'
            and row.get('script_version') == COLOC_SIGMA_RECORDED_VERSION):
        return None
    try:
        return float(row.get('coloc_threshold_value', ''))
    except (TypeError, ValueError):
        return None
