# core/ - CODE Directory

> **This is a CODE directory.** Core algorithms for 2D slice analysis.

## What This Is

The algorithmic heart of the sliceatlas package. Contains detection, colocalization, visualization, IO, and configuration modules.

## Key Files

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `config.py` | Path configuration, sample name parsing | `DATA_DIR`, `SampleDirs`, `parse_sample_name()`, `get_sample_dir()` |
| `detection.py` | Nuclei detection (StarDist + z-score peak finding) | `NucleiDetector`, `detect()` |
| `colocalization.py` | Signal colocalization analysis | `ColocalizationAnalyzer`, `classify_positive_negative()` |
| `visualization.py` | QC figure generation | `save_all_qc_figures()` |
| `io.py` | Image loading (ND2, TIFF) | `load_image()`, `extract_channels()` |
| `quantification.py` | Region-based quantification | Counts per atlas region |
| `registration.py` | Slice-to-atlas registration | Atlas alignment |
| `preprocessing.py` | Image preprocessing | Background subtraction, normalization |
| `boundaries.py` | Atlas region boundaries | Boundary detection and overlay |
| `atlas_utils.py` | Atlas coordinate utilities | AP position lookup |
| `insets.py`, `inset_detection.py` | High-magnification inset handling | Inset region detection |
| `image_utils.py` | Low-level image operations | Cropping, resizing, normalization |
| `channel_model.py` | Multi-channel data model | Channel metadata |
| `cellpose_backend.py` | Cellpose segmentation backend | Alternative to StarDist |
| `deepslice_wrapper.py` | DeepSlice integration | Automatic AP position estimation |
| `elastix_registration.py` | Elastix-based registration | Non-rigid registration |

## Rules

1. **config.py imports from mousebrain.config**: The canonical path source is `mousebrain.config`. This config imports `SLICES_2D_DIR`, `SLICES_2D_SUBJECTS`, etc. from there. Do not add independent path detection logic.
2. **SampleDirs defines folder names**: All pipeline subfolder names (`0_Raw`, `3_Detected`, `4_Quantified`, etc.) are defined in `SampleDirs`. Always use these constants, never hardcode folder names.
3. **parse_sample_name()** handles both compact (`E02_01_S13_DCN`) and full (`ENCR_02_01_S13_DCN`) formats. Test both when modifying.
4. **Detection defaults**: `z_cutoff=8.0` (conservative - catches only clear H2B-mCherry nuclei). `prob_thresh=0.5`, `nms_thresh=0.4` for StarDist.
5. **Channel convention**: Channel 0 = 488nm green (eYFP), Channel 1 = 561nm red (H2B-mCherry nuclear). These are defined as `RED_IDX`/`GREEN_IDX` in batch scripts.
