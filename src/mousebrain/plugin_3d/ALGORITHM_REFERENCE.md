# 3D Cleared Brain Algorithm Reference

> Auto-generated reference for the 3D cleared brain analysis pipeline algorithms.
> For the most current parameter values, see the source files referenced in each section.

## Pipeline Overview

The 3D cleared brain pipeline processes whole-brain light-sheet microscopy images of iDISCO+-cleared mouse brains. Raw data arrives as Imaris `.ims` files containing two channels: a signal channel (c-Fos, GFP, etc.) showing sparse bright cells, and a background/autofluorescence channel showing overall tissue structure.

The pipeline has six sequential stages:

1. **Organization** creates the folder structure for a brain
2. **Extraction & Analysis** converts .ims to TIFF stacks, identifies channels, crops the tissue
3. **Registration** aligns the brain to the Allen Mouse Brain Atlas
4. **Detection** finds cell candidate positions using a ball filter
5. **Classification** filters candidates through a trained neural network
6. **Region Counting** assigns classified cells to atlas regions

All parameters and results are recorded to a calibration run tracker (CSV) for reproducibility.

---

## 1. Extraction & Channel Identification

**Source:** `util_Scripts/2_extract_and_analyze.py` (v1.4.0)

### What It Does

Extracts multi-channel volumetric data from Imaris `.ims` files into individual TIFF slices per channel, identifies which channel is signal vs. background, and optionally crops the Y-axis to remove spinal cord.

### IMS Extraction

The `.ims` file is read using `imaris-ims-file-reader`. Each Z-plane of each channel is saved as an individual 16-bit TIFF file (`Z0000.tif`, `Z0001.tif`, ...). This produces two channel folders (`ch0/`, `ch1/`) under `1_Extracted_Full/`.

### Channel Identification

The pipeline needs to know which channel contains sparse cell signal and which contains diffuse tissue autofluorescence. It distinguishes them by analyzing texture:

1. **Compute sparsity** — for each channel, threshold at the 95th percentile and measure what fraction of pixels exceed it. The signal channel (c-Fos, GFP) has isolated bright spots, so fewer pixels above the threshold (sparsity < 0.05). The background channel has smoother, more uniform brightness.
2. **Compute local variance** — the signal channel has higher local variance due to punctate labeling.
3. The channel with lower sparsity is assigned as signal; the other as background.

This assignment is saved to `metadata.json` and used by all downstream scripts.

### Y-Axis Crop Detection (Auto-Crop)

**Note:** Auto-crop is disabled by default because it sometimes cuts into brain tissue. Manual cropping via `mousebrain --crop BRAIN_NAME` is recommended instead.

The algorithm detects where the brain transitions to spinal cord by analyzing tissue width:

1. Create a maximum-intensity projection across Z (one 2D image)
2. For each Y row, measure the width of tissue (number of pixels above background)
3. Compute the gradient of width vs. Y position
4. Find the Y position where width drops sharply — this is the brain-to-cord transition
5. Crop everything below that Y position

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CROP_DROP_THRESHOLD` | 0.85 | Width must drop to 85% of plateau to trigger |
| `MIN_BRAIN_HEIGHT_FRACTION` | 0.5 | Keep at least 50% of Y dimension |
| `MIN_GRADIENT_FOR_CROP` | -5 px/row | Minimum gradient to detect transition |
| `CAMERA_PIXEL_SIZE` | 6.5 µm | Andor Neo/Zyla sCMOS pixel size |
| `DEFAULT_ORIENTATION` | `"iar"` | inferior-anterior-right |

### Voxel Size Calculation

Voxel sizes are computed from the `.ims` file metadata:

- **XY voxel size** = `CAMERA_PIXEL_SIZE / magnification` (e.g., 6.5 / 1.625 = 4.0 µm)
- **Z voxel size** = Z-step from microscope metadata (typically 3–5 µm)

These are stored in `metadata.json` and passed to brainreg and cellfinder.

### Known Limitations

- **Channel identification assumes two channels.** Multi-channel (3+) images require manual assignment.
- **Auto-crop uses width only.** It cannot distinguish brainstem (which should be kept) from cord (which should be removed) when they have similar width.
- **IMS files on network drives are slow.** Extraction of a typical brain (~500 Z-planes, 2 channels) from a network-mounted `.ims` file can be very slow.

---

## 2. Registration (brainreg)

**Source:** `util_Scripts/3_register_to_atlas.py` (v2.0.2)

### What It Does

Registers the cropped brain to the Allen Mouse Brain Atlas using brainreg, producing a spatial mapping between sample coordinates and atlas coordinates. After registration, every voxel in the sample can be assigned to a named brain region.

### Registration Method

brainreg performs affine + freeform (non-linear) registration using NiftyReg. The background/autofluorescence channel is used for registration because it shows overall tissue morphology (signal channels are too sparse).

The algorithm:
1. Downsamples the sample image to match atlas resolution (10 µm isotropic)
2. Runs affine registration (translation, rotation, scaling, shearing)
3. Runs freeform deformation to handle local tissue distortions
4. Produces forward and inverse deformation fields
5. Maps the atlas annotation volume into sample space

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `atlas` | `allen_mouse_10um` | Allen Mouse Brain Atlas at 10 µm resolution |
| `orientation` | `iar` | inferior-anterior-right (how the sample was imaged) |
| `voxel_z` | From metadata | Z voxel size in µm (typically 3–5 µm) |
| `voxel_xy` | From metadata | XY voxel size in µm (typically 3–5 µm) |
| `n_free_cpus` | 50 | CPUs to leave free (Windows limits to 61 workers max) |

### Orientation Convention

The three-letter orientation code describes the anatomical direction of each image axis (Z, Y, X):

| Letter | Meaning |
|--------|---------|
| `i` | Inferior (ventral → dorsal along Z) |
| `a` | Anterior (front → back along Y) |
| `r` | Right (left → right along X) |

The default `iar` means: Z goes ventral-to-dorsal, Y goes anterior-to-posterior, X goes left-to-right.

### Registration Archiving

Previous registrations are never overwritten. When re-registering:

```
3_Registered_Atlas/
├── registered_atlas.tiff      ← Current registration
├── brainreg.json
└── _archive/
    ├── 20241216_143052/       ← Previous attempt #1
    └── 20241216_160215/       ← Previous attempt #2
```

### Registration Approval

After registration, the user must review QC images and explicitly approve before cell detection can proceed. Approval creates a `.registration_approved` marker file. Script 4 will refuse to run without this file.

### Known Limitations

- **Tissue damage confounds registration.** Large lesions, missing tissue, or surgical damage cause local registration errors that propagate to region assignment.
- **Orientation must be correct.** If the orientation code is wrong, registration will fail silently — the atlas will be mapped to the wrong axes.
- **10 µm atlas resolution.** Small structures below 10 µm are not resolved in the atlas. Cell assignments near region boundaries have inherent ~10 µm uncertainty.

---

## 3. Cell Candidate Detection (cellfinder)

**Source:** `util_Scripts/4_detect_cells.py` (v1.0.1)

### What It Detects

Bright cell-like objects in the signal channel that may be labeled neurons (c-Fos+, GFP+, etc.). These are *candidates* — the classification step (Script 5) determines which are real cells vs. artifacts.

### Detection Method: Ball Filter

cellfinder uses a 3D ball (sphere) filter approach:

1. **3D Gaussian smoothing** of the signal volume
2. **Ball filter** — a spherical structural element is convolved with the volume. The filter enhances objects that match the expected soma size (defined by `ball_xy_size` and `ball_z_size`). Objects much larger or smaller than the ball are suppressed.
3. **Thresholding** — voxels with filter response exceeding `n_sds_above_mean_thresh` standard deviations above the local mean are marked as candidates
4. **Connected component labeling** — adjacent bright voxels are grouped into candidate objects
5. **Centroid extraction** — each candidate gets a (z, y, x) coordinate

### Detection Presets

Four presets provide tested parameter combinations:

| Preset | ball_xy | ball_z | soma_diameter | threshold | Use Case |
|--------|---------|--------|---------------|-----------|----------|
| `sensitive` | 4 | 10 | 12 | 8 | Catches more cells, more false positives |
| `balanced` | 6 | 15 | 16 | 10 | Good default starting point |
| `conservative` | 8 | 20 | 20 | 12 | Fewer false positives, may miss dim cells |
| `large_cells` | 10 | 25 | 25 | 10 | Motor neurons, Purkinje cells, etc. |

### Paradigm-Best Settings

If a brain has previously been processed with the same imaging paradigm (same magnification and Z-step, e.g., `1p625x_z4`), the tracker can retrieve the best-performing detection parameters from that paradigm. This avoids re-tuning for each new brain of the same type.

The paradigm is extracted from the brain name: `349_CNT_01_02_1p625x_z4` → paradigm `1p625x_z4`.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ball_xy_size` | 6 (balanced) | Ball filter radius in XY plane (pixels). Larger = detects larger somas |
| `ball_z_size` | 15 (balanced) | Ball filter radius in Z (pixels). Typically 2–3x `ball_xy` due to anisotropic voxels |
| `soma_diameter` | 16 (balanced) | Expected soma diameter (pixels). Used for connected component grouping |
| `n_sds_above_mean_thresh` | 10 (balanced) | Detection threshold in standard deviations. Higher = fewer, brighter detections |
| `n_free_cpus` | 2 | CPUs to leave free during detection |

### Known Limitations

- **Anisotropic voxels require separate XY and Z parameters.** Light-sheet data typically has ~4 µm XY but ~4 µm Z, but the ball filter treats them independently.
- **Ball filter assumes spherical cells.** Elongated neurons (e.g., pyramidal cells) may be split into multiple candidates or missed entirely.
- **Threshold coupling.** `ball_xy_size`, `ball_z_size`, and `threshold` interact — changing one often requires adjusting the others. The presets provide tested combinations.
- **No local adaptation.** A single global threshold is applied. Regions with high autofluorescence (e.g., hippocampus) may generate more false positives than quiet regions.

---

## 4. Cell Classification (cellfinder ResNet50)

**Source:** `util_Scripts/5_classify_cells.py` (v1.0.0)

### What It Does

Takes the candidate positions from detection and classifies each as a real cell or a false positive (artifact, noise, blood vessel, etc.) using a trained convolutional neural network.

### Classification Method

For each candidate cell at position (z, y, x):

1. **Cube extraction** — a 50×50×20 voxel cube (height × width × depth) is extracted from the signal volume, centered on the candidate position. A corresponding cube is extracted from the background channel.
2. **Voxel resampling** — the cubes are resampled to a standard voxel size of (5.0, 1.0, 1.0) µm (Z, Y, X) so the network sees a consistent physical scale regardless of the original imaging resolution.
3. **Network inference** — the signal and background cubes are passed through a ResNet50 architecture that outputs a binary classification: `CELL` or `NO_CELL`.
4. **Output** — classified cells are saved to `cells.xml`, rejected candidates to `rejected.xml`.

### Network Architecture

| Property | Value |
|----------|-------|
| Architecture | ResNet50 |
| Input | 50×50×20 cube (signal) + 50×50×20 cube (background) |
| Network voxel sizes | (5.0, 1.0, 1.0) µm (Z, Y, X) |
| Output | Binary: CELL or NO_CELL |
| Model format | `.keras` (preferred) or `.h5` (weights only) |
| Inference batch size | 32 |

### Model Selection

Models are stored in a shared models directory. The selection priority is:

1. **Explicit** — user specifies `--model path/to/model.keras`
2. **Paradigm-best** — tracker retrieves the best model for the brain's imaging paradigm
3. **Default** — most recent model in the models directory (sorted by folder name/timestamp)

### Paradigm-Best Model

Like detection settings, the best-performing classification model can be stored per imaging paradigm. When processing a new brain with paradigm `1p625x_z4`, the system checks if a proven model exists for that paradigm and uses it automatically.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cube_height` | 50 px | Cube size in Y |
| `cube_width` | 50 px | Cube size in X |
| `cube_depth` | 20 px | Cube size in Z |
| `network_voxel_sizes` | (5.0, 1.0, 1.0) µm | Standard voxel sizes for network input (Z, Y, X) |
| `network_depth` | `"50"` | ResNet depth (ResNet50) |
| `batch_size` | 32 | Inference batch size |
| `n_free_cpus` | 2 | CPUs to leave free |

### Known Limitations

- **Training data bias.** The network performs best on tissue types similar to its training data. Tissue with unusual autofluorescence patterns or novel cell types may have higher error rates.
- **Cube edge effects.** Cells near the volume boundary get zero-padded cubes, which can confuse the network. Edge cells have slightly higher rejection rates.
- **Background channel required.** The network expects both signal and background cubes. If only one channel is available, classification quality degrades.
- **Model format.** `.keras` files contain the full model (architecture + weights). `.h5` files contain only weights and assume the standard ResNet50 architecture. Using `.h5` with a modified architecture will silently produce wrong results.

---

## 5. Regional Cell Counting

**Source:** `util_Scripts/6_count_regions.py` (v1.0.1)

### What It Does

Takes classified cell positions and assigns each to a named brain region using the registered atlas, then produces per-region counts.

### Region Assignment

For each classified cell at position (z, y, x) in sample space:

1. Use the registration's inverse deformation field to map the position to atlas space
2. Look up the atlas annotation value at that position
3. Map the annotation ID to a region name using the Allen Mouse Brain Atlas ontology

### Coordinate Transformation

The registration (Script 3) produces deformation fields that map between sample and atlas coordinates. brainglobe-utils handles the coordinate transformation, accounting for:

- Voxel size differences between sample and atlas
- Orientation differences (sample `iar` → atlas standard orientation)
- Non-linear tissue deformation from the freeform registration

### Output Files

| File | Contents |
|------|----------|
| `region_counts.csv` | Per-region cell counts (region name, cell count, hemisphere) |
| `cells_in_regions.csv` | Per-cell data with region assignments |
| Summary statistics | Total cells, top regions, lateralization |

### Known Limitations

- **Registration quality propagates.** If registration is poor in a region, all cell assignments in that region are wrong.
- **Point-based assignment.** Each cell is assigned based on its centroid coordinate. Large cells spanning a boundary are assigned to whichever region contains their center.
- **Atlas resolution.** The 10 µm atlas cannot resolve structures smaller than ~10 µm. Cells near fine boundaries (e.g., cortical layers) may be assigned to adjacent regions.

---

## 6. Calibration Run Tracking

**Source:** `util_Scripts/experiment_tracker.py`

### What It Records

Every analysis run across all pipeline stages is logged to a CSV (`calibration_runs.csv`) with 49+ columns:

| Field Category | Examples |
|---------------|----------|
| **Identity** | `exp_id`, `brain`, `created_at`, `hostname`, `username` |
| **Hierarchy** | `project`, `cohort`, `subject`, `imaging_params` |
| **Detection params** | `det_ball_xy`, `det_ball_z`, `det_soma_diameter`, `det_threshold`, `det_preset` |
| **Classification params** | `class_model_path`, `class_cube_size`, `class_batch_size` |
| **Registration params** | `reg_atlas`, `reg_orientation`, `reg_voxel_z`, `reg_voxel_xy` |
| **Results** | `det_cells_found`, `class_cells_found`, `class_rejected`, `duration_seconds` |
| **Paths** | `input_path`, `output_path` |
| **Quality** | `status`, `rating`, `marked_best`, `notes` |

### Brain Name Parsing

Brain names encode experimental hierarchy:

| Format | Example | Parsed As |
|--------|---------|-----------|
| Standard | `349_CNT_01_02_1p625x_z4` | brain=349, project=CNT, cohort=01, subject=02, mag=1.625x, zstep=4µm |
| Paradigm | `1p625x_z4` | Imaging paradigm (magnification + Z-step) |

The paradigm is used to retrieve best settings across brains with identical imaging parameters.

### Paradigm-Best System

When a user marks a run as "best" for a brain, the tracker records the detection parameters and classification model associated with that paradigm. When a new brain with the same paradigm is processed, these proven settings are offered as defaults.

This implements a form of transfer learning at the parameter level — settings optimized on one brain of a given type are automatically suggested for the next brain of the same type.

---

## Physical Context

### Typical Image Properties

| Property | Typical Value |
|----------|--------------|
| Image size | 2000–4000 × 2000–4000 × 300–800 (X × Y × Z) |
| Channels | 2 (signal + background) |
| Bit depth | 16-bit (uint16) |
| Voxel size XY | 3–5 µm |
| Voxel size Z | 3–5 µm |
| Signal channel | c-Fos, GFP, tdTomato |
| Background channel | Tissue autofluorescence |
| Typical cell diameter | 10–25 µm |

### Common Imaging Paradigms

| Paradigm | Magnification | Z-Step | XY Voxel | Notes |
|----------|---------------|--------|----------|-------|
| `1p625x_z4` | 1.625× | 4 µm | 4.0 µm | Standard, near-isotropic |
| `1p9x_z3p37` | 1.9× | 3.37 µm | 3.42 µm | Higher resolution |

---

## Version History

| Component | Version | Key Changes |
|-----------|---------|-------------|
| Organization | v1.0.0 | Folder structure creation |
| Extraction | v1.4.0 | Gradient-based crop detection, channel identification by sparsity+texture |
| Registration | v2.0.2 | brainreg with archiving, QC generation, approval gating |
| Detection | v1.0.1 | Python API (replaces deprecated CLI), 4 presets, paradigm-best lookup |
| Classification | v1.0.0 | ResNet50 with .keras support, paradigm-best model lookup |
| Region Counting | v1.0.1 | brainglobe-utils coordinate transformation, per-region CSV |
| Tracking | — | 49-column CSV with paradigm-best retrieval |

---

*This document describes the algorithms as implemented in code. For the source of each parameter value, see the files referenced at the top of each section.*
