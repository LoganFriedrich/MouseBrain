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

**The threshold line is always a fold change.** The fold-change histogram and
the intensity scatter draw the threshold as "times the background". For the
`background_mean` method, which classifies at *background + sigma x std*, the
batch script (section 6) draws the line at that cutoff divided by the
background, so the line sits exactly where its cells were split. (With sigma 0
that is 1.0x, the background itself.)

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

**What `coloc_threshold_value` means depends on the row.** Read it together
with `coloc_threshold_method` and `script_version`:

| `coloc_threshold_method` | `script_version` | `coloc_threshold_value` is |
|---|---|---|
| `background_mean` | `1.1.0` | the **sigma**: a cell is positive when its signal is above background mean + sigma x background std |
| `background_mean` | anything older (empty, `1.0.0`) | not trustworthy: the napari widget wrote its fold-change box here even for this method. The batch never uses such a row |
| `fold_change` (and `area_fraction`) | any | a **fold change**: times the background |
| `absolute` | any | an intensity value |
| `percentile` | any | a percentile of the cells' intensities |

`coloc_background_method` is `gmm+local` (or another name followed by
`+local`) when the widget's local-background box was ticked: a block-wise
background, not the global one of the same name. Colocalization rows written
by the widget or the batch from `script_version` `1.1.0` on also carry
`soma_dilation=...; bg_exclusion_dilation=...` in `notes`.

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

## 6. Batch Colocalization (`batch_encr`): choosing its numbers and running it

**Source:** `sliceatlas/batch/batch_encr.py`

The batch script runs ONE fixed method on every ND2 file of a project:
threshold + LoG nucleus detection, GMM background, `background_mean`
classification with a sigma threshold, measured in the green channel. What
you choose is six numbers: the sigma threshold, the background percentile,
the soma dilation, the background-exclusion dilation and the two detector
settings (`threshold_fraction`, `log_threshold`).

### Two different things: the batch's numbers and the approved method

- **The batch's numbers** are what it runs with. They come from the list
  below; marking a run best in BrainSlice is one way to choose two of them.
- **The approved method** is what the analysis registry judges every result
  of this batch (analysis `ENCR_Detection`) against. It comes from the
  approved-method file (`approved_method.json` under the registry root, or the
  file named by `MOUSEBRAIN_APPROVED_METHOD`):
  - if the file is in the per-analysis form and names `ENCR_Detection` under
    `"analyses"`, that entry is the batch's approved method;
  - otherwise it is the shared method: `"default"` in the per-analysis form,
    or the whole file in the plain form;
  - without any file it is the built-in method.

  A result made with any number that differs from it is registered as
  **stale**, and shows so in `mousebrain-registry --stale` and in any database
  that pulls the registry. The batch takes its untyped numbers from, and
  compares itself with, this same method -- never a different one.
  If the file gives `ENCR_Detection` the entry `null` ("no approved method"),
  the banner says `Approved:  none ...`, untyped numbers come from the
  built-in defaults, and the batch's results are never reported stale.

Marking a run best does **not** change the approved method. If you mark a run
whose sigma or percentile differs from it, the batch uses the run's numbers,
prints `[!] NOT the approved method` with the differences, and registers its
results as stale. To make those numbers approved, write them into the
approved-method file (next section) and the batch's next results are current.

### Making numbers the approved method

1. Run the batch once with `--dry-run` (below). Its banner lists every number
   it would use; under `[!] NOT the approved method` it lists each one that
   differs, as `this run ...` and `approved ...`.
2. Open (or create) `approved_method.json` in the registry folder
   (`<pipeline root>/Registry/`). It is a plain text file holding one JSON
   object. The parameter names are: `detection`, `threshold_fraction`,
   `log_threshold`, `log_decision_tree`, `size_filter_um`, `colocalization`,
   `sigma_threshold`, `soma_dilation`, `background`, `background_percentile`,
   `bg_exclusion_dilation`.

   - If the file already has `"default"` or `"analyses"` at its top (the
     per-analysis form), put the numbers for this batch under
     `"analyses"` -> `"ENCR_Detection"` (this changes only this batch), or
     into `"default"` (this changes every analysis that follows the shared
     method and has no entry of its own):

     ```json
     {"default": {"...": "..."},
      "analyses": {
        "ENCR_Detection": {"detection": "threshold+log", "threshold_fraction": 0.2,
          "log_threshold": 0.005, "log_decision_tree": true,
          "size_filter_um": [10, 25], "colocalization": "background_mean",
          "sigma_threshold": 0.5, "soma_dilation": 6, "background": "gmm",
          "background_percentile": 10, "bg_exclusion_dilation": 50}}}
     ```

   - Otherwise (the plain form: the parameters alone) edit the values in
     place, or change the file to the form above to give this batch its own:

     ```json
     {"detection": "threshold+log", "threshold_fraction": 0.2,
      "log_threshold": 0.005, "log_decision_tree": true,
      "size_filter_um": [10, 25], "colocalization": "background_mean",
      "sigma_threshold": 0.5, "soma_dilation": 6, "background": "gmm",
      "background_percentile": 10, "bg_exclusion_dilation": 50}
     ```

   The values in the examples only illustrate the format; the size range in
   particular is the open question at the end of this section. The full
   description of both forms is in `docs/pipeline/ANALYSIS_REGISTRY.md`
   ("The approved method").

3. Put in the `this run` value of every line the banner listed, save, and run
   `--dry-run` again: the `[!] NOT the approved method` block is gone when the
   two agree. An unreadable file stops the batch with a message naming it.

Results registered before the change are judged against the new approved
method too; any made with other numbers now show as stale. The registry's
header picks the change up at the batch's next registration, or at once with
`mousebrain-registry --name ENCR_Detection --refresh-approved`.

### Where each number comes from

For each number, the first of these that gives one wins:

| Order | Source | Gives |
|-------|--------|-------|
| 1 | What you type on the command line (`--threshold`, `--bg-percentile`, `--soma-dilation`, `--dilation`) | that number, even when it equals the default |
| 2 | A colocalization run of **this project** marked best in the tracker (skipped with `--ignore-tracker`) | sigma threshold and background percentile only |
| 3 | The approved method of `ENCR_Detection` (see above) | every number |
| 4 | Built-in defaults (sigma 0, percentile 10, soma 6 px, exclusion 50, threshold_fraction 0.20, log_threshold 0.005) | every number |

The banner printed at the start lists every number with its source in square
brackets, e.g. `sigma_threshold=0.5 [marked-best run col_...]`.

### Choosing the sigma and percentile in the napari widget

1. In BrainSlice, run colocalization on an image of the project with
   **Single Channel** mode, threshold method **background_mean**, background
   method **gmm**, the local-background box **unticked**, and a **soma
   dilation above 0**.
2. In the **Run History** panel, click the run, then **Mark Selected as Best**.
3. A message says whether the batch will use it, and for which project: the
   tracker's project for that run (read from the sample name), or any batch
   whose files include that sample. A batch of another project ignores it. If it says *will NOT use*,
   it gives the reason (another method, another channel, soma dilation 0, a
   run logged before runs recorded their sigma, ...): run again as in step 1
   and mark the new run.

**Not the same measurement.** The widget measures each cell's green signal in
a disk around its nucleus (the disks of neighbouring cells may overlap), using
the widget's own detection settings. The batch measures a ring around each
nucleus that leaves the nucleus out and is never shared with a neighbour, and
detects with its own fixed settings. A sigma chosen in the widget is therefore
applied by the batch to a related, not identical, measurement. A run with soma
dilation 0 measured the nucleus itself and is never used.

**Per project.** A marked run counts for a batch when its slice is one of the
files the batch processes, or the tracker files it under the batch's project.
Marks on other projects' slices are ignored (the batch prints how many).

**One mark per sample.** Marking a run clears the previous best of the same
sample only. If marked runs of different samples of the project carry
**different** numbers, the batch stops before processing anything and lists
them with their samples. To settle it, open each sample whose mark you did not
mean, select its starred run in Run History and click **Unmark Best**. (Or
type both `--threshold` and `--bg-percentile`, or use `--ignore-tracker`.)
Nothing else counts: an unmarked run, however recent, never changes the batch.
Run History lists the 20 newest completed runs of the open sample, followed by
every run of that sample marked best (whatever its age), so a starred run can
always be found and unmarked there. If the panel cannot be used (no napari at
hand), type both `--threshold` and `--bg-percentile`, or use
`--ignore-tracker`.

The soma dilation and exclusion dilation of a run are written in its notes
(`soma_dilation=...; bg_exclusion_dilation=...`) for reference; the batch does
not take them from the run. Type them, or set them in the approved method.

### Running it

```
conda activate <your MouseBrain environment>
python -m mousebrain.plugin_2d.sliceatlas.batch.batch_encr --project PROJECT_A --dry-run
python -m mousebrain.plugin_2d.sliceatlas.batch.batch_encr --project PROJECT_A
```

`--dry-run` prints the banner and the file list and changes nothing. Read the
banner before the real run.

### What it records

- **Tracker** (`calibration_runs.csv`): per file, a detection row with the
  pixel-area range the detector applied to that image and, in its notes, the
  diameter range and detector settings; and a colocalization row with the
  sigma threshold actually used.
- **Registry** (`ENCR_Detection`): per sample, the counts and the method
  parameters **this run used** (not simply the approved method). When any of
  them differs from the approved method, the banner says
  `[!] NOT the approved method` and lists the differences; those entries show
  as stale.
- **A sample with no nuclei** gets a current entry with 0 nuclei and no files.
  Its earlier entry's files are moved to `_archived/`, and the earlier record
  is kept inside the new entry under `superseded`, with
  `invalidated_reason` saying why and `outputs` naming where its files now
  are in `_archived/`. The end-of-batch summary lists these
  samples.
- **A sample whose run fails** (file cannot be read, tracker file locked by
  another program, any other error) changes nothing in the registry: an
  error is not a result, so its earlier entry stays current. The console
  prints `[!] not retired: ...` for it, and the end-of-batch summary lists
  every such sample under `Failed, earlier result kept as current -- re-run
  these`. Its row in `summary.csv` has status `error` and the message. Fix the
  cause (close the tracker file, reconnect the share) and run the batch again.

### Open question: the nucleus diameter range

The batch detects nuclei 10-25 um across. The built-in approved method says
8-25 um. Until a lab settles which is approved (by setting `size_filter_um`
in its approved-method file, or by a code change), every batch run with the
built-in approved method prints the `NOT the approved method` warning for
`size_filter_um` and its entries show as stale. That is the record being
honest, not a failed run.

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
