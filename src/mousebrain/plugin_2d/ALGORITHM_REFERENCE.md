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
| **Detection params** | `det_backend`, `det_preprocessing` (every setting, see below), `det_model`, `det_prob_thresh`, `det_nms_thresh`, `det_scale`, `det_min_area`, `det_max_area` |
| **Colocalization params** | `coloc_background_method`, `coloc_threshold_method`, `coloc_threshold_value` |
| **Particle analysis params** | `pa_detection_channel`, `pa_threshold_value`, `pa_measurement_channel`, `pa_min_area`, `pa_max_area`, `pa_min_circularity`, `pa_bg_method` |
| **Results** | `det_nuclei_found`, `det_raw_count`, `det_removed_by_size` / `_border` / `_morphology` / `_confidence`, `coloc_positive_cells`, `coloc_negative_cells`, `coloc_positive_fraction`, `pa_particles_found`, `pa_particles_raw`, `pa_bg_value`, `pa_mean_intensity` |
| **Paths** | `input_path`, `output_path`, `measurements_path` (`labels_path` and `region_counts_path` exist but nothing writes them: the window keeps nucleus labels in memory only, and a region count's folder is in its atlas-fit row) |
| **Quality** | `status`, `rating`, `marked_best`, `notes` |

The file lives at `<MouseBrain_Pipeline>/2D_Slices/2_Data_Summary/calibration_runs.csv`.

**Reproducing a detection from its row.** `det_backend` says which method ran
(`stardist`, `cellpose`, `threshold`, `threshold+log`) and `det_preprocessing` holds
every setting the napari window ran it with, as JSON text (for example
`{"backend": "threshold", "threshold_method": "otsu", "opening_radius": 2, ...}`):
set the controls of the Detect tab to those values to run it again. `det_prob_thresh`
and `det_nms_thresh` are StarDist's numbers and are blank for a backend that does not
use them. When the run finishes, the row also gets how many nuclei were found before
filtering (`det_raw_count`) and how many each filter removed (`det_removed_by_size`,
`_border`, `_morphology`, `_confidence`) -- the same numbers the window shows under
the result. Detections run by the batch keep their settings in `notes`.
Rows written before this change have only the StarDist numbers (`0` for a threshold
run) and cannot be reproduced from the row.

**Which detection a colocalization measured.** A colocalization row's `parent_run`
is the `run_id` of the detection whose nuclei it measured -- the detection on screen
when Colocalize was clicked -- for the napari window (single- and dual-channel) as for
the batch. To see the detection settings behind a colocalization, find the row whose
`run_id` is that `parent_run`. `parent_run` is blank when those nuclei have no row of
their own (the tracker could not be opened when they were detected). Colocalization
rows the window wrote before this change have a blank `parent_run`.

**Dual-channel colocalization counts.** A dual-channel run (`coloc_signal_channel` is
`dual`) puts each nucleus in one of four classes, each counted in its own column:
`coloc_dual_cells` (positive in both channels), `coloc_ch1_only_cells` (positive in the
first, nuclear channel -- red -- only), `coloc_ch2_only_cells` (positive in the second
channel -- green -- only) and `coloc_neither_cells`. Its `coloc_positive_cells`,
`coloc_negative_cells` and `coloc_positive_fraction` are blank: those are
single-channel counts. The Run History shows a dual run as `<n> dual`. **Older dual
rows** (written before this change) hold the "both" count in `coloc_positive_cells`
and the "neither" count in `coloc_negative_cells`; never read those two columns of a
`dual` row as a positive/negative count.

**Particle analysis background.** `pa_bg_method` says where the background came from:
`manual_roi` -- the value in the Background box (measured from drawn rectangles or
typed in); `percentile_50` -- the box was 0, so the median of the particles' own
pixels was used. (Older rows say `manual_roi` for both.) `pa_particles_raw` is the
number of particles before the size and circularity filters, `pa_particles_found`
after them, and `pa_mean_intensity` the average of the particles' mean intensities.

A program that asks the tracker to record a result under a name that is not one of
its result columns gets an error and nothing is written, instead of the value being
dropped without a word.

### What a run's status means

| `status` | Meaning |
|----------|---------|
| `started` | The run is going on right now. A row is written only once the run's inputs have been checked, so a run refused by a check ("Run detection first", "No signal channel data available" ...) leaves no row at all. |
| `completed` | The run finished and its result is in the row. |
| `failed` | The run did not finish, or finished but its result could not be written. The `notes` column says why (for example `batch stopped at colocalize: RuntimeError: ...`, `colocalization failed: ...`, `colocalization result could not be handled: ...`, `result could not be recorded: ...`). |

`failed` is the only word for a run that did not finish. Older rows written by the
batch may say `error`; that means the same thing (writing `error` now records `failed`).

A row that is already `completed` is never changed to `failed` by a later problem.
The batch analyzes each image in two runs, detection and then colocalization. The
colocalization row is set to `completed` only after its measurements CSV and its QC
pictures have been saved. If the colocalization fails, or saving its CSV or pictures
fails, the **colocalization** row becomes `failed` (the reason in `notes`) and the
finished detection row stays `completed`.

A row stays `started`:

- while its run is going on;
- when the program was killed outright (closed from Task Manager, the machine
  switched off) -- nothing can record anything then;
- when the napari window was closed while a run was still going, so its result
  never arrived;
- when neither the result nor the failure could be written, because
  `calibration_runs.csv` itself refused the write (see the next paragraph). The
  napari window then said so in its status bar. The batch printed
  `[SliceTracker] [!] Could not mark run ... failed`, and that sample's error in
  `summary.csv` ends with `tracker row <run id> could not be marked failed and
  still reads 'started'`. The batch itself goes on with the next sample.

Any `started` row older than the current session is one of these; treat it as failed.

**When the napari window cannot write a finished result to the tracker** -- most
often because `calibration_runs.csv` is open in a spreadsheet program. This is the
same for every kind of run the window records: nuclei detection (with or without
insets), colocalization (single- and dual-channel) and particle analysis.

- The status bar first says `Saving the result to the calibration tracker...`. The
  window does not respond while it waits: at most about 12 seconds (10 for another
  writer's lock, 2 for the file to be released). The batch waits much longer (up to
  5 minutes for the lock, 30 seconds for the file), because nobody is waiting on its
  window.
- If the write still cannot be made, the status bar ends with
  `[!] not saved to the tracker (<the reason>); its row in the tracker still says 'started' -- is calibration_runs.csv open elsewhere?`.
  The window does not try a second write to mark the row failed: the file would
  refuse it the same way and the window would freeze again. (When the write failed
  for some other reason than the file, the row is marked failed instead and the
  line says `run marked failed`.)
- The result stays on screen (the nuclei layer, the colocalization colors, the
  particle count). For a colocalization the measurements CSV, which is saved before
  the tracker is written, is on disk. But the run will not appear as completed in
  the Run History. Close the spreadsheet and run it again.

**When the napari window cannot write a NEW colocalization row** (the file refuses
the new line, most often because it is open in a spreadsheet program), the
colocalization does not start: a message box and the status bar say
`Colocalization not started: the calibration tracker refused its row (<the reason>)`,
and the Colocalize button can be clicked again at once. Close the spreadsheet and
click Colocalize again. (Detection behaves the same way: it says `Failed to start
detection` and does not run.) A run with no row could never be found, reloaded or
marked best later, so it is not run unrecorded.

**When something goes wrong while the window handles a colocalization's result**
(single- or dual-channel: a value missing from the result, a layer that cannot be
drawn), the status bar says `Colocalization error: <the reason>` (or `Dual
colocalization error: ...`), the run's row becomes `failed` with
`... result could not be handled: <the reason>` in `notes`, and the Colocalize
button works again.

**When the dual-channel colocalization cannot save its measurements CSV** (a full
disk, a folder that cannot be written), the status bar ends with
`[!] measurements CSV not saved (<the reason>); run marked failed`. The run is not
recorded as completed, because a run with no CSV on disk cannot be reloaded from the
Run History. (Single-channel colocalization still records such a run as completed
with no measurements file; the reason is printed in the console.)

### When runs are NOT being logged

If the tracker cannot be opened, nothing is logged, and you are told:

- **In the napari window:** a red line under the status bar reads
  `[!] Runs will NOT be logged to the calibration tracker: <the reason>`. It stays
  there for as long as the window is open. Fix the reason and reopen the window
  before tuning anything you want to keep a record of.
- **In the batch:** every sample prints the same `[!] Runs will NOT be logged ...`
  line. The sample is still analyzed and its CSV and pictures are still written.

The most common reason is that mousebrain does not know where the lab's data lives
(`CONNECTOME_ROOT` is not set): the tracker then refuses to open rather than
creating a new, empty tracker in a temporary folder, where runs would look logged
and be lost. Check with `mousebrain --paths`.

### One analysis at a time in the napari window

The window runs one analysis at a time: Detect Nuclei, Colocalize, Run Quantification
and Run Particle Analysis. While one is going (from the click until its result has
been shown and recorded), clicking another one does nothing except the status bar
saying, for example,
`Colocalization not started: detection is still running. Wait for it to finish, then click again.`
Wait for the first one's result, then click again. Nothing has been logged for the
refused click, so there is no row to clean up.

WHY: each run writes its result into its own row of the tracker. When two ran at
once, the window used to write the first one's result into the second one's row: a
detection's nucleus count landed in a colocalization row (which then said
`completed` with no colocalization result) and the detection row stayed `started`.
And a colocalization started while a detection is going would measure the nuclei the
detection is about to replace.

Rows written before this change may show that mix-up: a colocalization row with a
`det_nuclei_found` value, or a particle-analysis row with one. Treat such a row as
not trustworthy and run that analysis again.

### Several windows and the batch at once

The napari window and the batch can write the tracker at the same time, on one
machine or several. Every write first takes a lock file,
`calibration_runs.csv.lock`, beside the tracker and removes it when done (well under
a second). A writer that finds the lock waits for it, and gives up with a message
naming who holds it: the batch after 5 minutes, the napari window after 10 seconds
(see above). A lock left behind by a program that crashed is taken over
automatically after 2 minutes; until then the window's writes give up and say
`not saved to the tracker`. Never delete the lock by hand while anything may be
writing.

A change to the file is written to a temporary file next to it
(`calibration_runs.csv.tmp-...`) and then swapped in, so a crash part-way leaves the
previous tracker whole. Columns the file has that this version of the code does not
know are kept, and a row is always written in the file's own column order, so files
written by older or newer versions stay readable. If the tracker is ever found with
a row that has more values than the header has columns, writing stops with a
message asking for the file to be repaired: it is never rewritten in a way that
would lose those values.

Opening a tracker written by an older version adds the newer columns to its header
once: that is one rewrite, under the lock and through a temporary file, with every
value kept. It also happens on read-only paths, such as the batch looking up the
tuned parameters, so a read of an old tracker can briefly take the lock.

### Run Loading

Previous runs can be reloaded from the widget's Run History panel. The measurements CSV is read back from disk, and the colocalization visualization is re-displayed in napari. This allows:
- Comparing different parameter settings on the same image
- Reviewing results produced by batch scripts or other users
- Loading a "best" run marked by a previous reviewer

The Run History lists the runs of the sample on screen only: the name in the
Sample ID box, or the image's file name when the box is empty, matched exactly
(slice `..._S1` does not list the runs of `..._S10`). **Load Selected Run** refuses
a run made on another sample and says which sample it belongs to: its cells are
positions in that sample's image, and drawn over this one they would look like
(and be counted as) this slice's result. Open that sample's image first.

### Sample Name Parsing

Sample IDs are automatically parsed to extract hierarchy fields:

| Format | Example | Parsed As |
|--------|---------|-----------|
| Compact | `E01_01_S12` | project=ENCR, cohort=01, subject=01, slice=12 |
| Full | `ENCR_01_01_S12` | project=ENCR, cohort=01, subject=01, slice=12 |
| Generic | `anything_else` | Underscore-split with pattern matching |

---

## 6. Batch Colocalization (`batch_encr`): choosing its numbers and running it

**Source:** `sliceatlas/batch/batch_encr.py`

The batch script runs ONE fixed method on every ND2 file of a project:
threshold + LoG nucleus detection, GMM background, `background_mean`
classification with a sigma threshold, measured in the green channel. What
you choose is the nucleus diameter range (micrometers) and six numbers: the
sigma threshold, the background percentile, the soma dilation, the
background-exclusion dilation and the two detector settings
(`threshold_fraction`, `log_threshold`).

### Two different things: the batch's numbers and the approved method

- **The batch's numbers** are what it runs with. They come from the list
  below; marking a run best in BrainSlice is one way to choose two of them.
- **The approved method** is what the analysis registry judges the batch's
  results against: the approved method the registry holds for the batch's
  analysis, `<project>_Detection` -- one per project, named after the
  project code the batch was run for (`--project`, e.g. `ENCR_Detection` for
  a project coded `ENCR`); the banner's `Registry:` line names it. It is set in the approved-method file
  (`approved_method.json` under the registry root, or the file named by
  `MOUSEBRAIN_APPROVED_METHOD`); without that file it is the built-in method.
  The batch asks the registry for exactly this method, so its banner compares
  with what the registry will compare with. A result made with any number
  that differs from it is registered as **stale**, and shows so in
  `mousebrain-registry --name <project>_Detection --stale` and in any database that
  pulls the registry.

Marking a run best does **not** change the approved method. If you mark a run
whose sigma or percentile differs from it, the batch uses the run's numbers,
prints `[!] NOT the approved method` with the differences, and registers its
results as stale. To make those numbers approved, write them into the
approved-method file (next section) and the batch's next results are current.

### Making numbers the approved method

**First: whose approved method you are editing.** The approved-method file
can hold one SHARED method, used by every analysis that follows it, and a
separate entry per analysis (the per-analysis form, explained in
`docs/pipeline/ANALYSIS_REGISTRY.md`, "Giving each analysis its own approved
method"). Which one the batch (`<project>_Detection`) is judged by:

- the file names `<project>_Detection` under `"analyses"` -> that entry, and only
  that entry is the batch's;
- otherwise (no file, a file holding the parameters alone, or a per-analysis
  file that does not name it) -> the SHARED method, which is also the
  approved method of every other analysis that follows it -- the ROI analysis
  (`<project>_ROI_Analysis`) included, unless the file names that
  one too.

Editing the shared method therefore changes what those other analyses are
judged against: every existing entry of theirs is compared with the new
numbers, so the ones made with the old numbers show as **stale**
(`mousebrain-registry --name <project>_ROI_Analysis --stale`, and in any
database that pulls the registry). The ROI counter records, in each of its
entries, the parameters of the detection entry it counted against
(`docs/pipeline/ROI_COUNTING.md`), so its counts go stale together with the
detection they rest on -- as long as both analyses are judged by the same
approved method. Give the batch its own entry (step 2) and the ROI analysis
the same one, or every ROI count shows as stale; `brainslice-roi batch`
warns when the two differ.

How the banner tells you: whenever it lists a difference from the approved
method (`[!] NOT the approved method`) -- whatever the numbers came from:
typed, a marked run or the defaults -- it adds `[!] On this registry that
file is the approved method of EVERY analysis it holds` when the batch is
judged by the shared method. A banner with no difference listed says
nothing either way; then look in the file for an `"analyses"` entry named
`<project>_Detection`.

If you do not want to change the other analyses, give the batch its own
entry instead of editing the shared method (step 2).

1. Run the batch once with `--dry-run` (below). Its banner lists every number
   it would use; under `[!] NOT the approved method` it lists each one that
   differs, as `this run ...` and `approved ...`.
2. Open (or create) `approved_method.json` in the registry folder
   (`<pipeline root>/Registry/`). The batch's method is one JSON object of
   name -> value, with these names: `detection`,
   `threshold_fraction`, `log_threshold`, `log_decision_tree`,
   `size_filter_um`, `colocalization`, `sigma_threshold`, `soma_dilation`,
   `background`, `background_percentile`, `bg_exclusion_dilation`. Example:

   ```json
   {"detection": "threshold+log", "threshold_fraction": 0.2,
    "log_threshold": 0.005, "log_decision_tree": true,
    "size_filter_um": [10, 25], "colocalization": "background_mean",
    "sigma_threshold": 0.5, "soma_dilation": 6, "background": "gmm",
    "background_percentile": 10, "bg_exclusion_dilation": 50}
   ```

   The values in the example only illustrate the format: put in the numbers
   your lab chose for its tissue -- the diameter range (`size_filter_um`,
   smallest first, in micrometers) especially, because the batch detects
   with exactly that range (see "The nucleus diameter range" below). To make it
   the batch's own approved method, leaving every other analysis as it is,
   put that object under `"analyses"` as the entry for `<project>_Detection`:

   ```json
   {"default":  { ...the shared method, as before (or leave "default" out
                  for the built-in parameters)... },
    "analyses": {"<project>_Detection": { ...the object above... }}}
   ```

   If the file holds the parameters alone (the old form), move them under
   `"default"` first so the other analyses keep their method. Writing the
   object as the whole file (or as `"default"`) instead makes it the
   approved method of every analysis that follows the shared method (see
   "First: whose approved method you are editing" above).

   Use only the names listed. A name the batch has no setting for (any other
   name) is never used by the batch, so every batch entry would differ from
   the approved method and show as stale; the banner then says
   `The batch has no setting for <name>`.

3. Put in the `this run` value of every line the banner listed, save, and run
   `--dry-run` again: the `[!] NOT the approved method` block is gone when the
   two agree. An unreadable file stops the batch with a message naming it.
   Then refresh the header of the analysis you changed (and, if you edited
   the shared method, of every analysis that follows it):
   `mousebrain-registry --name <project>_Detection --refresh-approved --stale`.

Results registered before the change are judged against the new approved
method too; any made with other numbers now show as stale.

### Where each number comes from

For each number, the first of these that gives one wins:

| Order | Source | Gives |
|-------|--------|-------|
| 1 | What you type on the command line (`--threshold`, `--bg-percentile`, `--soma-dilation`, `--dilation`) | that number, even when it equals the default |
| 2 | A colocalization run of **this project** marked best in the tracker (skipped with `--ignore-tracker`) | sigma threshold and background percentile only (its dilations must match; see below) |
| 3 | The approved method | every number |
| 4 | Built-in defaults (sigma 0, percentile 10, soma 6 px, exclusion 50, threshold_fraction 0.20, log_threshold 0.005) | every number |

The banner printed at the start lists every number with its source in square
brackets, e.g. `sigma_threshold=0.5 [marked-best run col_...]`.

The nucleus diameter range follows its own, shorter list: `--diameter MIN MAX`
typed on the command line, else `size_filter_um` of the approved method. A
marked run never supplies it, and there is **no built-in range** (see "The
nucleus diameter range" below).

### Choosing the sigma and percentile in the napari widget

1. In BrainSlice, run colocalization on an image of the project with
   **Single Channel** mode, threshold method **background_mean**, background
   method **gmm**, the local-background box **unticked**, and a **soma
   dilation above 0**.
2. In the **Run History** panel, click the run, then **Mark Selected as Best**.
   Marking a run clears the mark of the sample's previous best run; the
   message names that run (`The mark on run ... was cleared`).
3. A message says whether the batch will use it. If it says *will NOT use*,
   it gives the reason (another method, another channel, soma dilation 0, a
   run logged before runs recorded their sigma, ...): run again as in step 1
   and mark the new run.

   If the batch will NOT use the run you selected while the sample already
   has a marked run the batch DOES use, you are asked first (`Replace a mark
   the batch uses?`), with that run named. **No** leaves everything as it
   was. **Yes** marks the new run and clears the old mark, and the batch then
   takes its sigma and percentile from elsewhere (another sample's mark, or
   the approved method).

**Not the same measurement.** The widget measures each cell's green signal in
a disk around its nucleus (the disks of neighboring cells may overlap), using
the widget's own detection settings. The batch measures a ring around each
nucleus that leaves the nucleus out and is never shared with a neighbor, and
detects with its own fixed settings. A sigma chosen in the widget is therefore
applied by the batch to a related, not identical, measurement. A run with soma
dilation 0 measured the nucleus itself and is never used.

**Per project.** A marked run counts for a batch when its slice is one of the
files the batch processes, or the tracker files it under the batch's project.
Marks on other projects' slices are ignored (the batch prints how many).

**One mark per sample.** Marking a run clears the previous best of the same
sample only. If marked runs of different samples of the project carry
**different** numbers (sigma, percentile, or the dilations they were made at),
the batch stops before processing anything and lists them with their samples.
To settle it, open each sample whose mark you did not mean, select its starred
run in Run History and click **Unmark Best**. (Or type both `--threshold` and
`--bg-percentile`: the batch then lists the disagreeing runs, uses neither,
and its banner says `Tracker: marked-best runs disagree; none used, because
--threshold and --bg-percentile were typed`. Or use `--ignore-tracker`.)
Nothing else counts: an
unmarked run, however recent, never changes the batch. Run History lists the
20 newest completed runs of the open sample **and every marked run of it**,
however old, so a mark can always be found and cleared there.

**The dilations must match.** A run's soma dilation and exclusion dilation are
written in its notes (`soma_dilation=...; bg_exclusion_dilation=...`). The
batch does not take them from the run -- they come from the command line or
the approved method -- but a sigma is chosen for one measurement, so it checks
them whenever it takes a number (the sigma or the percentile) from the marked
run: when the batch would run at a different soma or exclusion dilation than
the marked run was made at, it **stops before processing anything** and names
both values, e.g. `soma_dilation: marked-best run col_... was made at 3, the
batch would run at 6 [approved method]`. To go on, do one of these:

- type the run's value (`--soma-dilation 3`, `--dilation 30`) to run at the
  dilations the sigma was chosen at;
- type the batch's own value to keep it -- the batch then runs and repeats the
  difference in its banner as `[!] soma_dilation: the marked-best run was made
  at 3; this batch runs at 6 (typed on the command line)`;
- mark a run made at the batch's dilations instead; or
- use `--ignore-tracker`.

A marked run whose notes do not record both dilations is never used.

When you type **both** `--threshold` and `--bg-percentile`, the marked run
supplies nothing, so its dilations are not checked and the batch runs with
what you typed (the same rule that lets typing both settle disagreeing marked
runs).

### Running it

```
conda activate <your MouseBrain environment>
python -m mousebrain.plugin_2d.sliceatlas.batch.batch_encr --project PROJECT_A --dry-run
python -m mousebrain.plugin_2d.sliceatlas.batch.batch_encr --project PROJECT_A
```

`--dry-run` prints the banner and the file list and changes nothing. Read the
banner before the real run. If it stops with `no nucleus diameter range is
set`, write `size_filter_um` into the approved-method file (see "The nucleus
diameter range" below), or add `--diameter MIN MAX` to run once at a range you
type.

### What it records

- **Tracker** (`calibration_runs.csv`): per file, a detection row with the
  pixel-area range the detector applied to that image and, in its notes, the
  diameter range and detector settings; and a colocalization row with the
  sigma threshold actually used.
- **Registry** (`<project>_Detection`): per sample, the counts and the method
  parameters **this run used** (not simply the approved method): exactly the
  eleven names listed under "Making numbers the approved method", nothing
  else. When any of them differs from the approved method, or the approved
  method names something the batch has no setting for, the banner says
  `[!] NOT the approved method` and lists the differences; those entries show
  as stale.
- **A sample with no nuclei** is a result and is registered like one: a
  current entry with 0 nuclei, no files, a positive fraction and background of
  `null` (nothing was measured), and the parameters the run used. If the
  sample already had a current entry, every file in the sample's own
  registry folder (`exports/<project>_Detection/<sample>/`) is first moved to an
  `_archived/<date_time>/` folder inside it -- not only the files the entry
  lists, but also any other file there (an older batch copied extra
  overlays and histograms it never listed); the `_archived/` folder itself
  is left alone. The earlier entry's record is replaced in `registry.json`,
  but not lost: the registry's log (`logs/<project>_Detection.log`) gets one
  `invalidate_entry` line for the retired entry carrying its counts, method
  hash and parameters (see "Where a retired entry's counts are kept" below),
  then a `register` line for the new one. This happens even when the earlier
  entry has no files to move (an earlier 0-nuclei entry lists none; a
  completed entry's files may have gone missing): it is still retired, and
  still gets its `invalidate_entry` line, before the new entry is written. The end-of-batch summary lists each such sample as
  `earlier registry entry (<n> nuclei) replaced by this run's 0 nuclei;`
  followed by `<k> file(s) moved to _archived/` when files were moved, or
  `no files to archive` when there were none (or the earlier entry was
  already retired, that is, no longer current).

  **If those files cannot be moved** (any ONE of them is open in another
  program, or the share drops out), the ones already moved are put back and
  nothing is replaced: the 0-nuclei result is NOT
  registered, the earlier entry stays current with its files where they
  were, and the summary says `[!] <sample>: this run found 0 nuclei, but the
  earlier registry entry (<n> nuclei) is KEPT CURRENT: its files could not be
  moved to _archived/`. Close whatever has the files open and run the batch
  again. WHY it is not registered anyway, and WHY every file in the folder
  moves and not only the listed ones: otherwise files of the earlier run
  would stay in the active folder beside an entry that is not theirs (or
  beside no current entry at all), and anything that copies that folder --
  an integrator that mirrors the registry does -- would take them for
  current results.

  **If the files were moved and the 0-nuclei result then cannot be written**
  (the registry file is open in another program, or the share drops out at
  that moment), the sample is left with NO current entry: the earlier one is
  retired, its files are in `_archived/`, and nothing replaced it. The
  summary says `[!] <sample>: earlier registry entry (<n> nuclei) was
  retired, <k> file(s) moved to _archived/ (or: no files to archive), but this run's 0-nuclei result
  could NOT be registered: <error>. The sample has no current entry now; run
  the batch again.` Running the batch again registers the 0-nuclei result.
- **A sample whose run fails**: the batch goes on with the next file. Its
  row in `summary.csv` has status `error` and the message, and the
  end-of-batch summary lists every failed file under `Errors:` as
  `<file>: [<sample|infrastructure> failure while <step>] <message>`. What
  happens to the sample's earlier registry entry depends on whose failure it
  was:

  | The failure | Examples | Earlier registry entry |
  |---|---|---|
  | **The sample's**: while taking the two channels out of the image (after the file has been read), detecting nuclei or measuring cells, a data error (`ValueError`, `KeyError` or `IndexError`) -- and some OTHER sample of the same run got past that same step | the image has the wrong number of channels or dimensions, the detector rejects it | **retired**: no longer current, and every file in the sample's own registry folder moved to an `_archived/<date_time>/` folder inside it, listed or not, as for a sample with no nuclei above (nothing is deleted). The summary says `earlier registry entry retired (no longer current; <k> file(s) moved to _archived/)`, or `no files to archive` when there were none (an entry with 0 nuclei has no files). |
  | **Infrastructure**: ANY error while reading the file from disk (`read`), and anything else at any step | the share cannot be reached, **the file is not recognized as an ND2** (empty or damaged), **the file's end is missing** (cut off, or still being copied), the tracker file is open in another program, the disk is full, memory runs out, **a missing library or the wrong environment** (an import error), a library or code defect that raises a type, attribute or runtime error | **kept current**, unchanged. The summary says `earlier registry entry kept current -- the failure was not shown to be the sample's`. |
  | **A data error at a step no other sample of the run got past** | the same software fault hit every file; or every sample that has nuclei failed while measuring | **kept current**. The summary says `earlier registry entry kept current -- no other sample of this run got past the '<step>' step, so the failure may be the software's, not the sample's`. |

  The `<step>` in an `Errors:` line is one of: `read` (reading the file from
  disk), `channels` (taking the two channels out of the image that was
  read), `detect` (detecting nuclei), `colocalize` (measuring cells),
  `tracker` (writing the tracker) and `save` (saving the outputs). Only
  `channels`, `detect` and `colocalize` can ever be a `sample` failure.

  Which samples count as "got past the step": for a failure while taking
  the channels out of the image or detecting nuclei, any sample of the run
  that finished, with or without nuclei. For a failure while measuring cells, only a sample that
  finished WITH nuclei -- a sample with no nuclei is never measured, so it
  shows nothing about that step.

  **If a retirement cannot be carried out** because one of the files in the
  sample's folder cannot be moved (it is open in another program, or the
  share drops out), the files already moved are put back, the entry stays
  current and the summary says `[!] <sample>: the
  sample's own re-run failed, but its earlier registry entry could not be
  retired and is STILL CURRENT`. Close whatever has the files open and run
  the batch again.

  WHY the split: the newest attempt on a sample that failed on the sample
  itself contradicts the older count, so the older count must not stand as
  current. A failure of the share, the tracker, the environment or the code
  says nothing about the sample, and retiring a good result because of it
  would drop it for no reason. So only errors that can say something about
  the image's data count as the sample's, and only when another sample of
  the same run shows the software gets past that step.

  Reading the file from disk is deliberately on the infrastructure side,
  whatever error it gives. The ND2 reader reports two different errors
  there: an `OSError` for a file it does not recognize as an ND2 at all, and
  a `ValueError` for a file that starts like an ND2 but whose end is missing
  -- a copy that was cut off, or one still under way. Both keep the entry. A
  file that is only half there says nothing about the sample, and neither
  error can be told from a share that drops out during the read. (The
  `ValueError` is one of the three data errors, which is why reading has a
  step of its own: at any later step the image is already in memory, so the
  file and the share can no longer be the cause.) The cost: a file that is
  truly damaged for good also keeps its earlier entry, and shows up as an
  `infrastructure failure while read` on every run until the file is
  replaced.

  What the rule cannot know: the three data errors are also what a defect
  in the code or in a library typically raises. A defect that hits every
  image is caught by the "another sample got past that step" condition. A
  defect that hits only SOME images, in a run where another sample got past
  the step, is taken for those samples' own failure and their earlier
  entries ARE retired. The earlier files are kept when that happens: they
  are in `_archived/`, and the registry's log records the retirement together
  with the retired counts and parameters. Running
  the batch again once the defect is fixed registers each sample again, and
  that registration OVERWRITES the retired entry's record in `registry.json`
  -- its counts and parameters are not kept there. If the re-run does not
  reproduce the earlier counts exactly (the batch's detection is not yet
  guaranteed to), the earlier counts are then no longer in the registry's
  record, but they are kept in two places: the per-cell values in the
  archived `measurements.csv`, and the counts themselves (with the method
  hash and parameters that produced them) in the registry's log -- see
  "Where a retired entry's counts are kept" below. One more
  consequence: a batch run on a single sample never retires its entry,
  because nothing else in the run shows the software works; run it together
  with a sample known to be good.

  **Before the first sample** the batch imports every library the reading,
  detecting and measuring steps need (the ND2 reader, scipy, scikit-image,
  scikit-learn, pandas, matplotlib). If one is missing or broken it stops
  with `[!] this environment cannot run the batch; no sample was attempted
  and no registry entry was changed`, names the library, and exits with
  status 1. Activate the environment MouseBrain is installed in (or install
  the named library) and run again.

  Either way, fix the cause and run the batch again: a successful re-run
  registers a new current entry. A retired entry's files stay in `_archived/`
  and the registry's log records the retirement with the retired counts.

  **Where a retired entry's counts are kept.** Every time an entry is
  retired, the registry writes one line for it to `logs/<project>_Detection.log`
  in the registry folder. Open that file in Notepad and search for the
  sample name; the line you want has the word `invalidate_entry` in it.
  The columns are separated by tabs: the time, the computer, the word
  `invalidate_entry`, the sample, the entry's key, and then, in braces, the
  retired record: `results` (`n_nuclei`, `n_positive`, `positive_fraction`,
  `background` ...), `method_hash`, `method_params` (every setting the run
  used), `source_files`, `registered_at`, `invalidated_at`, and
  `archived_to` (the `_archived/<date_time>` folder its files were moved
  to). A sample retired more than once has one such line per retirement;
  the latest is the last one in the file. (Entries retired before this line
  was added have only an `invalidate` line with a count, no record.)

  **If that log line cannot be written** (the log file is open in another
  program, the share drops out at that moment), the entry is still retired
  -- writing the log never stops the batch -- and the registry prints
  `[!] <key> was invalidated, but its record could not be written to
  logs/<project>_Detection.log (<error>). Its record is: {...}` with the same
  record as the log line would have held. That printed line is then the
  ONLY place the retired counts and parameters are kept (the next
  registration of the sample overwrites them in `registry.json`): copy it
  from the window into a text file beside the log before closing the
  window. Once the sample is registered again, nothing else holds them.
- **When the registry itself cannot be read or written** (its file is open
  in another program, the share drops out), the samples are still processed
  and their files and `summary.csv` are written. The end-of-batch summary
  says what did not reach the registry, one line per case:

  | Summary line | What happened | The sample's registry entry now |
  |---|---|---|
  | `[!] <sample>: completed, but this run's result was NOT registered (<error>)` (under `Completed:`) | the sample finished with nuclei, and writing its entry failed | whatever it was before this run: an earlier entry, if the sample has one, is still the current one |
  | `[!] <sample>: the registry could not be read or updated for this sample (<error>)` (under `Errors:`) | a sample that found no nuclei, or failed, and the registry could not be read or changed for it | as it was before this run, unless another line of the summary says otherwise |
  | `[!] Registry: could not be opened (<error>). NO result of this run was registered` (last line before `Summary:`) | the registry could not be opened when the batch started | every entry is as it was before this run |

  In each case fix the cause (close the program that has the registry file
  open, reconnect the share) and run the batch again. `Completed: <n>`
  counts samples that were processed, not entries that were written: a
  sample is registered only if none of these lines names it.

### The nucleus diameter range

Only nuclei whose diameter falls inside this range are counted, so it decides
what the batch counts. Which range is right depends on your tissue, stain and
magnification; it is a setting, and the code carries no range of its own.

- **Where it comes from:** `--diameter MIN MAX` if you type it (for one run),
  else `size_filter_um` in the approved method the batch is judged by (see
  "Making numbers the approved method"). Without an approved-method file the
  approved method is the registry's built-in one, and its `size_filter_um` is
  then the range used: write your own into the file to choose it.
- **What you see:** the banner's `Detection:` line names the range and its
  source, e.g. `nucleus diameter 12-30 um [approved method]`. Each image's
  tracker detection row records it in its notes, with the pixel-area range it
  became at that image's pixel size, and each registry entry records it as
  `size_filter_um`.
- **When nothing gives a range** (the approved method has no `size_filter_um`,
  or the approved-method file gives `<project>_Detection` no approved method at
  all) and `--diameter` was not typed, the batch stops before reading any
  image and says how to set it. Nothing is written.
- **When the range is unusable** (not two numbers, the larger first, or 0 or
  below) the batch stops the same way and names the value it found.
- **A typed range that differs from the approved one** is used, recorded, and
  listed under `[!] NOT the approved method`; those entries show as stale,
  like any other typed number.

Changing `size_filter_um` in the approved method makes every earlier entry
made with the old range show as stale; re-run those samples to bring them
up to date.

### Open question: recording the measurement region

The batch measures each cell's signal in a ring around its nucleus that
leaves the nucleus out and does not overlap a neighbor's ring. That choice
is fixed in the code and is NOT yet among the method parameters a registry
entry records, so two entries cannot be told apart by it. Adding a name for
it changes the method hash of every entry, and which name and which approved
value to use is for a lab to settle; until then nothing about the region is
recorded.

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
