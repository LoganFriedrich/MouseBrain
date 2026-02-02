# BrainSlice 2D Algorithm Reference

> Auto-generated reference for the BrainSlice 2D analysis pipeline algorithms.
> For the most current parameter values, see the source files referenced in each section.

## Pipeline Overview

The BrainSlice 2D pipeline analyzes fluorescence microscopy images of coronal brain sections. A typical workflow loads a multi-channel image (nuclear stain + signal channel), detects individual cell nuclei, measures signal intensity in each nucleus, classifies cells as positive or negative for the signal, and optionally assigns cells to atlas regions for regional quantification.

The pipeline has four sequential stages:

1. **Detection** identifies individual cell nuclei using a neural network
2. **Colocalization** measures signal intensity and classifies positive/negative
3. **Quantification** assigns cells to brain regions and counts per region
4. **Export** saves measurements, QC images, and human-readable reports

All analysis parameters are recorded in a calibration run tracker (CSV) for reproducibility.

---

## 1. Nuclei Detection (StarDist)

**Source:** `sliceatlas/core/detection.py`

### What It Detects

Individual cell nuclei in a fluorescence image (typically the red/nuclear channel, e.g., H2B-mCherry). Each detected nucleus receives a unique integer label and a polygon boundary.

### Detection Method: StarDist2D

StarDist is a neural network that predicts star-convex polygons for each object. Unlike watershed or thresholding approaches, it handles touching/overlapping nuclei well because each nucleus gets its own predicted polygon.

The algorithm:

1. **Normalizes** the image using percentile-based contrast stretching (1st to 99.8th percentile mapped to 0–1). This makes the detection robust to different exposure levels across images.
2. **Optionally downscales** the image (e.g., 0.5x) for speed/memory on very large images. Coordinates are scaled back afterward.
3. **Runs StarDist2D.predict_instances()** which returns a label image (each pixel assigned to a nucleus ID or 0 for background) and detection details (polygon vertices, probabilities).
4. **Filters by size** — removes objects smaller than `min_area` (noise, debris) or larger than `max_area` (clumps, artifacts).
5. **Optionally filters by circularity** — removes elongated objects that aren't round nuclei. Circularity = 4π × area / perimeter². A perfect circle = 1.0.

### Pretrained Models

| Model | Use Case |
|-------|----------|
| `2D_versatile_fluo` (default) | General fluorescence microscopy |
| `2D_versatile_he` | H&E stained histology |
| `2D_paper_dsb2018` | Data Science Bowl 2018 nuclei |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `'2D_versatile_fluo'` | Pretrained StarDist model |
| `prob_thresh` | 0.5 | Detection probability threshold. Higher = fewer detections, more confident. Lower = more detections, more false positives |
| `nms_thresh` | 0.4 | Non-maximum suppression IoU threshold. Controls how much overlap is allowed between adjacent detections |
| `scale` | 1.0 | Image downscaling factor. Use 0.5 for large images to reduce memory |
| `min_area` | 50 px | Minimum nucleus area in pixels. Objects smaller than this are removed as noise |
| `max_area` | 5000 px | Maximum nucleus area. Objects larger than this are removed as clumps |
| `min_circularity` | 0.5 | Optional shape filter. 1.0 = perfect circle, lower = more elongated allowed |
| `normalize_input` | True | Apply percentile normalization before detection |

### Fallback Strategies

| Situation | What Happens |
|-----------|--------------|
| CSBDeep normalization unavailable | Falls back to simple `(x - lo) / (hi - lo)` percentile normalization |
| Windows symlink error loading model | Tries 3 strategies: standard load → extracted directory → non-symlink directory |
| Downscaling causes shape mismatch | Pads or crops label image to match original dimensions |

### Known Limitations

- **Assumes round/convex nuclei.** StarDist uses star-convex polygons, so it may fail on highly irregular or elongated cell shapes.
- **Parameter coupling.** `prob_thresh` and `nms_thresh` interact — changing one may require adjusting the other for optimal results.
- **Single-channel input only.** Accepts 2D grayscale. Multi-channel images must be split before detection.

---

## 2. Colocalization Analysis

**Source:** `sliceatlas/core/colocalization.py`

### What It Does

Measures signal intensity (e.g., green/eYFP channel) within each detected nucleus and classifies each cell as "positive" (expressing the signal) or "negative" (background only). The critical challenge is determining what counts as "background" — this is done by analyzing the tissue itself, not the dark empty space around the brain.

### Step 1: Tissue Mask Estimation

The algorithm needs to distinguish brain tissue from empty slide/mounting medium. Since nuclei are only found in tissue, it uses the detected nuclei as seeds:

1. Start with a binary mask of all detected nuclei
2. Dilate (expand) this mask by `dilation_iterations` pixels in all directions
3. The dilated region approximates brain tissue

This prevents the background estimate from being contaminated by the dark regions outside the brain, which would make every cell look "bright" by comparison.

### Step 2: Background Estimation

Background is estimated from tissue pixels that are **outside** detected nuclei — the neuropil/extracellular space between cells. This represents the natural autofluorescence of the tissue.

Four methods are available, in order of statistical rigor:

#### GMM Method (Recommended, Default)

A **Gaussian Mixture Model** fits 1-component and 2-component models to the tissue intensity distribution, then selects the better model using the Bayesian Information Criterion (BIC).

- **If 2 components win:** The tissue has two populations — a dim "background" population and a brighter "signal bleed" population. The lower-mean component is taken as true background. This handles cases where some signal leaks into the neuropil.
- **If 1 component wins:** The tissue has uniform brightness — there's one clear background level. This is the simplest case.

**Confidence metric:** When 2 components are found, the separation between them is measured as `(bright_mean - dim_mean) / dim_std`. Higher separation = more confident background estimate.

| Separation | Confidence | Interpretation |
|------------|------------|----------------|
| > 2.0 | High | Groups clearly distinct |
| 1.0 – 2.0 | Moderate | Some overlap, check visually |
| < 1.0 | Low | Hard to separate, consider reviewing |

**Implementation details:** Subsamples to 200,000 pixels if the tissue region is larger (GMM on millions of pixels is slow). Uses fixed random seed (42) for reproducibility.

#### Percentile Method

Takes the Nth percentile of tissue-outside-nuclei intensity. Simple and fast but doesn't account for bimodal distributions.

#### Mode Method

Estimates the mode (most common value) of the tissue intensity histogram using 256 bins. Good for unimodal distributions but sensitive to bin width.

#### Mean Method

Takes the mean of tissue intensity after excluding the top 5% (bright outliers). Simple but sensitive to skewed distributions.

### Step 3: Intensity Measurement

For each detected nucleus, the algorithm measures:

| Metric | How Computed |
|--------|-------------|
| `mean_intensity` | Mean signal value within the nucleus mask |
| `median_intensity` | Median signal value (more robust to outliers) |
| `max_intensity` | Maximum pixel value in the nucleus |
| `integrated_intensity` | Sum of all pixel values (area × mean) |

These are computed using `skimage.measure.regionprops_table` for mean/max (fast, vectorized) and manual iteration for median/integrated (not available in regionprops).

### Step 4: Classification

Each cell is classified as positive or negative based on its mean intensity relative to background:

#### Fold Change Method (Default)

```
fold_change = mean_intensity / background
positive if fold_change >= threshold (default: 2.0)
```

In plain terms: a cell must be at least 2x brighter than the tissue background to count as positive.

#### Absolute Method

```
positive if mean_intensity >= threshold
```

Uses a fixed intensity cutoff regardless of background.

#### Percentile Method

```
positive if mean_intensity >= Nth percentile of all nuclei
```

Classifies the brightest N% of cells as positive. Useful when background estimation is unreliable.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `background_method` | `'gmm'` | Background estimation method: `'gmm'`, `'percentile'`, `'mode'`, `'mean'` |
| `background_percentile` | 10.0 | Percentile value (only used with `'percentile'` method) |
| `dilation_iterations` | 20 (analyzer) / 50 (widget) | How far to expand nuclei mask to estimate tissue region. Larger = more tissue included |
| `threshold_method` | `'fold_change'` | Classification method: `'fold_change'`, `'absolute'`, `'percentile'` |
| `threshold_value` | 2.0 | Threshold for classification (interpretation depends on method) |

### Fallback Strategies

| Situation | What Happens |
|-----------|--------------|
| No tissue pixels outside nuclei | Uses all tissue pixels (including inside nuclei) |
| No tissue pixels at all | Uses entire image |
| Too few tissue pixels for GMM (<100) | Falls back to percentile method |
| GMM fitting fails | Falls back to percentile method |

### Known Limitations

- **Background is tissue-wide, not local.** A single background value is used for the entire image. If tissue autofluorescence varies spatially (e.g., cortex vs. hippocampus), some regions may have inflated or deflated fold-change values.
- **Binary classification only.** Each cell is positive or negative — no intermediate confidence score per cell (though fold-change provides a continuous measure).
- **Dilation parameter is critical.** Too small = tissue mask doesn't cover actual tissue, background includes dark regions. Too large = tissue mask extends beyond brain, background is diluted by mounting medium.

---

## 3. Regional Quantification

**Source:** `sliceatlas/core/quantification.py`

### What It Does

Assigns each detected cell to a brain atlas region based on its centroid position, then counts positive and negative cells per region. Requires a registered atlas label image where each pixel's value corresponds to a brain region ID.

### Region Assignment Algorithm

For each cell:
1. Round the centroid (y, x) to the nearest integer pixel
2. Clamp to atlas image bounds (cells at the edge snap to the nearest valid pixel)
3. Look up the region ID at that pixel in the atlas label image
4. Map the region ID to a human-readable name via the atlas manager

Cells whose centroid falls on region ID 0 are labeled "Outside Atlas."

### Counting

Groups cells by region and computes:

| Metric | Computation |
|--------|-------------|
| `total_cells` | Count of all cells in region |
| `positive_cells` | Count where `is_positive == True` |
| `negative_cells` | `total - positive` |
| `positive_fraction` | `positive / total` |

Results are sorted by total cell count (descending).

### Hierarchical Aggregation

Can aggregate counts to parent regions in the atlas hierarchy. For example, summing all cortical sub-regions into "Isocortex." Controlled by the `level` parameter:
- `level=1`: Immediate parent region
- `level=2`: Grandparent region
- etc.

### Export Outputs

| File | Contents |
|------|----------|
| `{sample}_cells.csv` | Per-cell measurements with region assignments |
| `{sample}_regions.csv` | Per-region counts and fractions |
| `{sample}_summary.txt` | Summary statistics |

### Known Limitations

- **Centroid-only assignment.** A cell is assigned to whichever region its center pixel falls in. Large nuclei straddling a region boundary are assigned to one side only.
- **Hard boundary clamping.** Cells with centroids outside the atlas bounds are snapped to the edge, potentially being assigned to the wrong region.
- **Requires registered atlas.** If the atlas is poorly aligned to the tissue, region assignments will be systematically wrong.

---

## 4. Visualization & Reporting

**Source:** `sliceatlas/core/visualization.py`

### QC Outputs

The pipeline generates several quality-control visualizations:

| Figure | What It Shows | Key Things to Check |
|--------|---------------|---------------------|
| **Overlay** | Green channel with colored nucleus boundaries (lime=positive, red=negative) | Do green-outlined cells visually appear brighter? |
| **Annotated Overlay** | Same as overlay, plus yellow arrows pointing to the top 30 brightest positive cells with fold-change labels | Do the arrows point to genuinely bright cells? |
| **GMM Diagnostic** | Histogram of tissue intensity with fitted Gaussian curves (blue=background, red=signal bleed) | Do the curves fit the histogram well? Are they well-separated? |
| **Fold Change Histogram** | Distribution of fold-change values with threshold line; green bars=positive, red=negative | Is the threshold in a sensible place? |
| **Intensity vs Area** | Scatter plot of nucleus area vs. mean intensity, colored by classification | Are there suspicious clusters? |
| **Background Mask** | Tissue mask (blue) and excluded nuclei (red) overlaid on the green channel | Does the blue region cover actual tissue? |
| **ROI Summary** | Bar chart of positive/negative counts per manually-drawn ROI | Consistent across ROIs? |

### Human-Readable Report

A plain-text report (`qc_report.txt`) is generated explaining:
1. **What was done** — in plain English
2. **How background was determined** — with a marble-sorting analogy for GMM
3. **How cells were classified** — with concrete numbers (e.g., "a cell needed brightness ≥ 142.6 to count")
4. **Results** — counts, fractions, fold-change statistics
5. **How to verify** — step-by-step guide to reading each QC figure

---

## 5. Run Tracking

**Source:** `sliceatlas/tracker/slice_tracker.py`

### What It Records

Every analysis run is logged to a CSV (`calibration_runs.csv`) with:

| Field Category | Examples |
|---------------|----------|
| **Identity** | `run_id`, `sample_id`, `created_at`, `hostname` |
| **Hierarchy** | `project`, `cohort`, `slice_num` |
| **Detection params** | `det_model`, `det_prob_thresh`, `det_nms_thresh`, `det_scale`, `det_min_area`, `det_max_area` |
| **Colocalization params** | `coloc_background_method`, `coloc_threshold_method`, `coloc_threshold_value` |
| **Results** | `det_nuclei_found`, `coloc_positive_cells`, `coloc_negative_cells`, `coloc_positive_fraction` |
| **Paths** | `input_path`, `output_path`, `measurements_path`, `labels_path` |
| **Quality** | `status`, `rating`, `marked_best`, `notes` |

### Run Loading

Previous runs can be reloaded from the widget's Run History panel. The measurements CSV is read back from disk, and the colocalization visualization is re-displayed in napari. This allows:
- Comparing different parameter settings on the same image
- Reviewing results produced by batch scripts or other users
- Loading a "best" run marked by a previous reviewer

### Sample Name Parsing

Sample IDs are automatically parsed to extract hierarchy fields:

| Format | Example | Parsed As |
|--------|---------|-----------|
| Compact | `E02_01_S12` | project=ENCR, cohort=02, subject=01, slice=12 |
| Full | `ENCR_02_01_S12` | project=ENCR, cohort=02, subject=01, slice=12 |
| Generic | `anything_else` | Underscore-split with pattern matching |

---

## Physical Context

### Typical Image Properties

| Property | Typical Value |
|----------|--------------|
| Image size | 2048×2048 to 4096×4096 pixels |
| Channels | 2–4 (nuclear, signal, optional others) |
| Bit depth | 16-bit (uint16) |
| Nuclear stain | H2B-mCherry (red channel) |
| Signal of interest | eYFP (green channel) |
| Nucleus diameter | ~10–30 pixels |

### Intensity Ranges

| Channel | Typical Background | Typical Positive Signal |
|---------|-------------------|------------------------|
| Nuclear (red) | 200–500 | 1000–10000 |
| Signal (green) | 100–300 | 500–5000 |

These vary significantly by exposure, tissue type, and imaging conditions. The percentile normalization and fold-change classification are designed to handle this variation.

---

## Version History

| Component | Key Changes |
|-----------|-------------|
| Detection | StarDist2D with size/circularity filtering, Windows symlink workaround |
| Colocalization | Added GMM background estimation with BIC model selection and confidence metrics |
| Quantification | Centroid-based region assignment with hierarchical aggregation |
| Visualization | Added annotated overlay with arrows, GMM diagnostic plot, human-readable report |
| Tracking | Per-run CSV logging with measurements path for historical run reloading |

---

*This document describes the algorithms as implemented in code. For the source of each parameter value, see the files referenced at the top of each section.*
