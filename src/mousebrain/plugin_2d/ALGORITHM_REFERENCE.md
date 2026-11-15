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
| `{sample}_atlas_fit.json` | The atlas fit the counts were made with, or a note that no Align-tab fit was used, that a count is in progress (or never reported back), or that a failed run left the counts incomplete (see below) |

### The atlas fit is saved and logged with every count

**Source:** `sliceatlas/core/atlas_fit.py`

Each press of **Run Quantification** records what the region counts were made
with, so a set of counts can always be traced back to its atlas fit after
napari is closed. The rule is simple: **a fit file never claims a fit the
counts beside it were not made with.** When the window cannot keep that true
by writing the right file, it removes the wrong one and tells you.

**What happens, step by step, when an atlas is loaded in the Align tab:**

1. **When you press Run Quantification**, the fit is taken from the Align tab
   at the same moment as the atlas regions handed to the counting step. Before
   counting starts, any earlier `{sample}_atlas_fit.json` is replaced by a
   record with `"counts_status": "in_progress"` (it keeps the earlier record
   inside it, under `previous_record`), and a `registration` row is added to
   the tracker (`calibration_runs.csv`) with status `started`.
   Why the earlier file is replaced first: the counts are written while
   counting runs, but the finished fit file only once the window hears back.
   If napari is closed or crashes in between, the folder must not show new
   counts beside a file describing the earlier fit. **If you find a fit file
   that still says `in_progress`** after napari is closed, that count never
   reported back: the count files beside it may be from that run or the one
   before, and no fit is claimed for them. Press Run Quantification again.
   If the earlier fit file cannot be replaced or removed (usually because
   another program has it open), **nothing is counted**: the warning
   *"Quantification not run"* appears, the count files are left as they are,
   and you should close that program (or delete the file) and press Run
   Quantification again.
2. **When counting succeeds**, the counts are exported (if "Export to CSV" is
   on), then `{sample}_atlas_fit.json` is written in the same `4_Quantified`
   folder, replacing any earlier one, and the tracker row is set to
   `completed` with the file's path in `reg_transform_path`.
3. **When counting fails** (for example a results CSV is open in another
   program), the tracker row is set to `failed` and the status line shows the
   error. If the count files were not changed, the earlier fit file is put
   back exactly as it was. The export of the three count files (`_cells.csv`, `_regions.csv`,
   `_summary.txt`) is all-or-nothing: the new files are written under
   temporary names first, and the old ones are swapped out only when all three
   can be; if any one of them is locked, all three old files are put back
   exactly as they were. So in the normal case an earlier run's counts stay,
   with the fit file that describes them.
   As a safety net, the window compares the count files before and after the
   run. If a failed run did change one of them (something the all-or-nothing
   export should prevent), the fit file is replaced by a record with
   `"counts_status": "incomplete"` and `"atlas_fit_used": null`, the warning
   *"Counts incomplete"* appears, and you should close whatever has the files
   open and press Run Quantification again.

The tracker row carries the atlas (`reg_atlas`), orientation
(`reg_orientation`), method (`reg_method`, always `align_tab_affine`), the
slice's front-to-back position in micrometers for coronal slices
(`reg_ap_position`) and the fit as JSON in `notes`. It is written even
when export is switched off; the file is not, because with export off there is
no results folder.

To find out whether a logged count finished, read the row's `status` column:
`completed` (counted, and exported if export was on), `failed` (the count
failed), or `started` (the window was closed or crashed before the count
reported back). The `notes` JSON describes the fit only. Unlike the fit file
it has no `counts_status` and no `atlas_fit_used` field, because `notes` is
written once, before anything is counted, and is not changed afterwards.

**When the counts are made without an Align-tab fit** (no atlas loaded, or
the Align tab had no regions to give), any earlier `{sample}_atlas_fit.json` is
replaced -- once the counts are exported -- by a record with
`"atlas_fit_used": false` and `labels_source` saying what the cells were
assigned with instead: `stored_atlas_labels` (region labels the window already
held) or `none_all_cells_unassigned` (no regions: every cell is in region 0).
No tracker row is written in that case, as there is no fit to log.

Programs that call the `quantify_sample()` function directly with an output
folder get the same protection: the earlier fit file is replaced by an
in-progress record before the counts are written and by a record with
`"atlas_fit_used": false` and `"labels_source": "supplied_by_caller"` after
(the function cannot know how the labels it was given were placed). If the
earlier fit file cannot be replaced, nothing is exported and the call fails
with an error.

What a fit record holds: the atlas name and resolution, orientation, the
position-slider value (an atlas slice number), and the position, rotation and
scale from the Align tab's controls -- the values the counts really use,
however they were reached (sliders, DeepSlice, auto-align or an ABBA import all
end in those controls). `elastix_warp_on_screen` says whether an Elastix warp
was drawn over this image and slice when Run Quantification was pressed; the
counts do NOT use that warp either way (see Known Limitations). It is `true`
only when all of these hold: "Register Atlas (Elastix)" really warped the atlas
(if Elastix returned no transform, nothing was warped), the warped atlas was
actually drawn, it was run on the image and slice now shown, and nothing has
redrawn the overlay since. These controls redraw the unwarped atlas and so
make it `false`: the position slider, the orientation choice, the X/Y offset,
rotation and scale boxes (and the Reset button, when it changes one of them),
and Load Atlas. A redraw also removes the warped region outlines that "Show
region boundaries" drew with the warp, so no warped outline is left on top of
the unwarped atlas (the outlines come back the next time Elastix is run).
Loading an image or moving to another slice of a stack also
makes it `false`, because the warp was computed for a different image. The
opacity slider and the Show reference / regions / boundaries checkboxes do NOT
redraw anything: after them the warp is still on screen and is still recorded
as on screen.

Elastix takes a while. If the position slider or the orientation is changed,
another atlas is loaded, or another image or slice is shown while it is still
running, the finished warp belongs to an atlas slice or image that is no
longer selected: it is not drawn, the status line says "Registration discarded
... Press Register Atlas (Elastix) again", and the fit stays recorded as
having no warp on screen. If the warped atlas cannot be drawn when Elastix
finishes, the status line says so in red, the unwarped atlas is drawn again,
and no warp is recorded as on screen.

Every fit file written on success carries `"counts_status": "complete"`.

The file is written to a temporary file first and then swapped in, so a crash
while saving never leaves a half-written file. If the new file cannot be
written at all after the counts were exported, the previous fit file is
**removed** (it describes a different fit than the new counts): no fit file is
better than a wrong one.

**Warnings you may see:**

- *"Atlas fit not logged"* -- the calibration tracker could not be opened, or
  could not be written (most often because `calibration_runs.csv` is open in
  a spreadsheet program; the reason is in brackets). The counts are still
  made. With export on, the fit is still saved beside the counts -- only the
  tracker row is missing; with export off, it is not recorded anywhere, and
  the window says so. Close the spreadsheet (or fix the tracker setting) and
  press Run Quantification again if you need the tracker row.
- *"Atlas fit not saved"* -- `{sample}_atlas_fit.json` could not be written.
  If the message begins "The counts were made", the new counts are in the
  folder; if it begins "The count failed", they are not (see "Counts
  incomplete" below), and it was the record saying the folder is incomplete
  that could not be written. The message says whether the previous fit file was
  removed. If it says the previous file could NOT be removed, that file
  describes a different fit than the counts beside it: delete it by hand.
- *"Counts incomplete"* -- see step 3 above.
- *"Atlas fit not recorded"* -- the fit could not even be read from the Align
  tab. The counts are still made, and any earlier fit file beside them is
  removed before counting starts.
- *"Quantification not run"* -- see step 1 above: the earlier fit file could
  not be replaced, so nothing was counted.

### Known Limitations

- **Centroid-only assignment.** A cell is assigned to whichever region its center pixel falls in. Large nuclei straddling a region boundary are assigned to one side only.
- **Hard boundary clamping.** Cells with centroids outside the atlas bounds are snapped to the edge, potentially being assigned to the wrong region.
- **Requires registered atlas.** If the atlas is poorly aligned to the tissue, region assignments will be systematically wrong.
- **The Elastix warp is shown but not counted.** After "Register Atlas (Elastix)" the overlay shows the warped atlas regions, but Run Quantification assigns cells with the unwarped atlas placed by the Align tab's position, rotation and scale. The saved fit records whether a warp was on screen (`elastix_warp_on_screen`, above). To see on screen what the counts use, move the position slider, change the orientation, or change an X/Y offset, rotation or scale box: these redraw the unwarped atlas. The opacity slider and the Show checkboxes do not -- the warp stays on screen after them.
- **The overlay ignores the image's pixel size; the counts work in image pixels.** The atlas overlay is drawn without the image's micrometer scale, while counting places the atlas on image pixels. When the image's pixel size is not 1, what lines up by eye on screen is not exactly what is counted.

### DeepSlice position predictions (Align tab, "Detect All Positions (DeepSlice)")

**Source:** `sliceatlas/core/deepslice_wrapper.py`

DeepSlice suggests an atlas position for every image in the loaded folder. The
suggestions are kept in `deepslice_positions.json` in that same folder (with
DeepSlice's own output in `DeepSlice_Output/` beside it), so a second press of
the button on the same folder answers at once instead of running the network
again.

The saved file also lists every image it was made from: name, size and the
time it was last changed. **The saved suggestions are reused only when the
folder still holds exactly those images.** If an image was added, removed or
re-exported since, the button runs DeepSlice again on the whole folder and
replaces the file (the status line says "Images changed since the cached
predictions -- running DeepSlice again"). A `deepslice_positions.json` written
by an older version holds no such list; it is never trusted and is replaced the
first time the button is pressed.

**Where these files live is not settled.** They are written into the image
folder because that is where they have always been. Moving them under the
pipeline folder is waiting on a decision about where per-image and
folder-wide 2D outputs belong (the same decision as for the per-image analysis
folders), so do not rely on the location staying as it is. Note that a fresh
DeepSlice run may delete and recreate `DeepSlice_Output/` in the image folder
(it always does when the images had to be converted first -- ND2 or TIFF -- or
the folder is on a network drive; for a local folder of PNG/JPG images it
writes its files into the existing folder). Keep nothing of your own in it.

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
| **Atlas fit** (`registration` rows, written by Run Quantification: `started`, then `completed` or `failed`) | `reg_atlas`, `reg_orientation`, `reg_method`, `reg_ap_position`, `reg_transform_path` |
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
| Compact | `E01_01_S12` | project=ENCR, cohort=01, subject=01, slice=12 |
| Full | `ENCR_01_01_S12` | project=ENCR, cohort=01, subject=01, slice=12 |
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
