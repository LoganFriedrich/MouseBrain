# Cell Detection Optimization Workflow

This guide walks you through optimizing cell detection parameters for your tissue.

## Overview

The optimization loop:
1. Run detection with specific parameters
2. Classify the candidates
3. Count cells by region
4. Compare to published reference data
5. Adjust parameters based on results
6. Repeat

## Prerequisites

- Brain registered to atlas (Script 3 complete)
- Registration approved (`python util_approve_registration.py --brain YOUR_BRAIN`) -- the
  approval is for the registration that is there now; registering again needs a new one
- cellfinder installed

## Step 1: Run Detection

### Using cellfinder's napari widget (recommended for tuning)

1. Open napari
2. Load your cropped brain image (from `2_Cropped_For_Registration/ch0/`)
3. Go to `Plugins` > `cellfinder` > `Cell detection`
4. Adjust parameters in the widget:
   - **Soma diameter**: Expected cell body size in microns
   - **Ball filter XY/Z**: Spatial filter sizes
   - **Threshold**: Detection sensitivity
   - and the other four (ball overlap, soma spread, LoG sigma, max cluster)
5. Run on a small region first to test

Setup & Tuning (`Plugins -> BrainTools -> 3D: 2. Setup & Tuning`) does the same
with every run logged, all eight settings included. Its **Settled method (from
tracker)** button starts you from the method already settled for this imaging
setup, if there is one. When a run is right, select it and click **Mark as
Paradigm Best**: that makes it the settled method every brain imaged this way is
detected with (README, "The Settled Detection Method").

### Using the CLI (recommended for full runs)

```bash
# The settled method for this brain's imaging setup
python 4_detect_cells.py --brain YOUR_BRAIN

# A trial with some settings changed (logged as "custom")
python 4_detect_cells.py --brain YOUR_BRAIN --ball-xy 8 --threshold 9

# No settled method yet: give all eight
python 4_detect_cells.py --brain YOUR_BRAIN --ball-xy N --ball-z N --soma-diameter N \
    --threshold N --ball-overlap N --soma-spread N --log-sigma N --max-cluster N
```

### Parameter guidance

| Parameter | Effect when increased | Effect when decreased |
|-----------|----------------------|----------------------|
| soma_diameter | Catches larger cells, misses small | Catches smaller cells, more noise |
| ball_xy_size | Smoother filtering, misses small | More sensitive, more false positives |
| ball_z_size | Better for thick sections | Better for thin sections |
| threshold | Fewer cells, higher confidence | More cells, lower confidence |

## Step 2: Classify Cells

```bash
python 5_classify_cells.py --brain YOUR_BRAIN
```

This uses the trained network to filter detection candidates into real cells vs artifacts.
Step 5 writes the brain's **production** classification,
`5_Classified_Cells/cells.xml` -- the file step 6 counts.

### Trying a model in napari does not change production

Classification runs started from Setup & Tuning ("7. Classify" tab, "Run
Classification") are calibration runs. Each one is written to its own folder,
`2_Data_Summary/optimization_runs/<brain>/class_<date>_<time>/` (`cells.xml`,
`rejected.xml`), and that folder is the run's `output_path` in the tracker. The
brain's `5_Classified_Cells` is not touched, so trying a model can never replace
the classification the counts come from.

**To make a napari run the production classification** (promote it), use any
of these three buttons; each asks you to confirm and does the same thing:

- **Promote last run to production...** under "Run Classification" ("7. Classify" tab): the last
  run made in this window for the selected brain.
- **Promote selected to production...** under "Classification Runs (from
  Tracker)" (the "8. Class Compare" tab): select any completed run of this
  brain in the list -- from today or from an earlier session -- then click it.
  Click **Refresh** first if the list is empty.
- **Promote a run...** at the top of the "10. Results" tab: a list of this brain's
  classification runs; select one and click **Promote Selected to
  Production...**.

A promotion is refused, with the reason shown, when the run belongs to another
brain, is not a completed classification, has no `cells.xml`, or IS the
production classification already (a step 5 run's files are in
`5_Classified_Cells` itself; promoting them onto themselves would archive them
and leave the brain with no classification). What a promotion does:

1. The files currently in `5_Classified_Cells` (`cells.xml`, `rejected.xml`,
   `promoted_from.json`) move to `5_Classified_Cells/_archive/<date>_<time>/` --
   kept, not deleted.
2. The run's `cells.xml` and `rejected.xml` are copied into `5_Classified_Cells`,
   with `promoted_from.json` naming the run, its folder, who promoted it and when.
   The copies carry the time of the promotion, not the time the run was made.
   Why: the copies used to keep the run's own, older time, so counts made after
   the run but before the promotion looked newer than the classification and
   were never shown as stale.
3. The run's tracker row is tagged `promoted_to_production`.
4. Region counts made before the promotion are now older than the
   classification, so RUN_PIPELINE and the pipeline dashboard show them as stale
   and offer counting again. Run step 6 to update them.

Running step 5 again also replaces production (that is what step 5 is for); it
archives the files it replaces in the same way.

### Curating training data (Setup & Tuning, "9. Curate/Train" tab)

Curation means answering, candidate by candidate, "is this a cell?" (C = cell,
X = not a cell). Every answer is training data for a classifier, so the window
keeps the answers like this:

- **Each brain's answers are kept in that brain's own folder**, inside the pool
  for its imaging setup:
  `<models folder>/<imaging setup>/training_data/<brain>/`. In it:
  `curated_cells.xml` and `curated_non_cells.xml` (the answers),
  `cells/` and `non_cells/` (the training cubes, once exported) and
  `training.yml`. "Create Training Layers" points the window at the selected
  brain's folder and shows its saved answers in the layers
  "Train: Cells (n)" and "Train: Non-Cells (n)". After every answer those
  layers are redrawn from what is saved for the selected brain -- never
  another brain's points.
- **Points you place by hand in "Train: Cells" / "Train: Non-Cells"** (napari's
  add-points mode) are answers too. They are saved into the selected brain's
  answers before the next answer is saved, before "Create Training Layers"
  redraws the layers, before "Create Training Dataset" cuts cubes, and -- into
  the brain you are leaving -- when you choose another brain. Two cases are NOT
  saved, and a message "Points Not Saved" lists them by z/y/x: a point that
  already has the other answer (or is saved as both), and a point placed in
  both layers. Saving those would quietly change an answer you gave. To change
  the answer of a candidate, answer that candidate again with C or X. To change
  the answer of a point you placed by hand (it is not a candidate), select it
  in its training layer and click "Add Confirmed to Cells Bucket" or "Add
  Rejected to Non-Cells Bucket": the last answer wins and the point moves to
  the other file. If a training layer is recorded as
  another brain's and holds points that are not saved for that brain, nothing
  is saved from it and the layers are left as they are ("Another Brain's
  Points"); select that brain again to keep them.
- **Deleting a point from a training layer does NOT take its answer back.** It
  comes back at the next redraw. If you click "Create Training Dataset" while
  saved answers are missing from the layers, nothing is cut and a message
  "Saved Answers Missing From the Layers" says how many: click "Create
  Training Layers" to redraw them and export again, or, to change an answer,
  answer that candidate again with C or X (a point placed by hand: select it
  and use "Add Confirmed to Cells Bucket" / "Add Rejected to Non-Cells Bucket").
- **"Show Training Data in Napari"** only shows the cubes found on disk, in
  layers named "Cubes on disk: Cells (n)" / "Cubes on disk: Non-Cells (n)".
  Those layers are for looking only: they are never read as answers, never
  cut into cubes, and never redrawn by an answer.
- **Every answer is saved, or you are told it was not.** If "Create Training
  Layers" has not been clicked for the selected brain, the first C or X sets
  up that brain's folder by itself and saves the answer there. When an answer
  cannot be saved, a message "Answer Not Saved" says why, and the candidate
  is not marked as answered, so it is asked again.
- **Choosing another brain** (Setup tab) puts the "Training data path" line
  back to "(not set)", empties the two training layers (the last brain's
  answers stay saved in its folder) and makes the curation widget let go of
  its candidates, which were the last brain's points. Select the new brain's
  candidates layer and click "Use Selected Layer" again. Answers given on a
  points layer that is recorded as another brain's (detection-run layers carry
  their brain) are refused with a message. Other points layers, such as
  classification results, carry no brain: after a brain change, make sure the
  layer you choose was made for the selected brain.
- **A review carries on where it stopped.** When you choose a candidates
  layer ("Use Selected Layer"), every candidate that already has a saved answer
  for the selected brain counts as answered again (green = cell, red = not a
  cell), the layer line says how many ("..., 12 already answered"), and review
  starts at the first candidate without an answer. This holds after choosing
  the same layer again and after restarting napari. Candidates you skipped
  (S / Space) are not saved anywhere, so they are asked again. Nothing is
  restored for a points layer recorded as another brain's.
- **One answer per point, the last one counts.** Every C or X is saved at once.
  If you change an answer (C, then X on the same candidate), the point moves
  from one file to the other; answering the same way twice saves it once. The
  buttons that add the points of a selected layer to the cells or non-cells
  bucket work the same way, and refuse a layer recorded as another brain's.
  They save the points you SELECTED in that layer. With no point selected they
  ask first: "Save ALL N points of this layer as cells?" (or NOT cells), with
  how many of those points are now saved with the other answer and would be
  changed. The answer is No unless you click Yes; on No nothing is saved.
  Saying Yes with, for example, "Train: Non-Cells" chosen and "Add Confirmed
  to Cells Bucket" clicked turns every saved rejection of the brain into a
  cell, so read the numbers before you say Yes.
  "Send suspicious to training" (Pre-Filter tab) and "Merge XML batches" only
  add points that have no answer yet: they never change an answer you gave.
- **Merge XML batches** takes only the selected brain's batches. Batches of
  other brains, and the pool's own older files (`curated_cells.xml` directly in
  `training_data/`, which do not say which brain they came from), are left out
  and named in the message. A point that is a cell in one batch and a non-cell
  in another is not merged.
- **Answers saved by older versions** (in the brain's own
  `1_Brains/.../<brain>/training_data/` folder, or timestamped
  `curated_*_<date>_<time>.xml` batches) are not read automatically. To bring
  them in: select the brain, "Scan for XML Batches", tick that brain's batches,
  "Merge Selected into Training Layers". Points saved as both cell and
  non-cell in the old files are not merged; answer them again.
- **Files are never half-written.** Both answers files are written to a
  temporary file and swapped in. If a saved answers file cannot be read, the
  window says so ("Answer Not Saved" / "Saved Answers Unreadable") and changes
  nothing; move the damaged file aside before answering again.
- **"Create Training Dataset" cuts cubes only from the selected brain's own
  image.** It takes the selected brain's saved answers and its own Signal (and
  Background) layers. If the viewer holds only another brain's images it
  refuses ("No Image of This Brain"); load this brain's images first. Cubes
  are written to the brain's `cells/` and `non_cells/`. A cube that contradicts
  the answers -- its point now has the other label, or is saved as both -- is
  removed, and the message says how many files were removed; cubes of points
  that have no answer in the files (for example copied in by "Merge Selected ->
  Shared Pool") are kept. Points
  closer than half a cube (10 planes, 25 pixels) to the image edge are skipped
  and counted in the message.
- **Training from the pool** ("Shared: <imaging setup>" in the training
  sources table) trains on every brain folder's cubes: the pool's
  `training.yml` is rewritten to list each brain's `cells/` and `non_cells/`
  every time training starts or a brain is exported. Cubes left directly in
  the pool's own `cells/` and `non_cells/` by older versions are still listed;
  they carry no brain in their names, so which brain each came from cannot be
  checked. "Merge to shared pool" copies a brain's cubes into that brain's
  folder in the pool, and "Start fresh" deletes the cubes of the pool and of
  every brain folder in it but never the answers files -- export again to cut
  them anew.

### Training a model from Setup & Tuning ("9. Curate/Train" tab, "Start Training")

The training runs in its own process (`scripts/utils/util_train_model.py`);
napari stays usable and the console shows its progress.

- The widget names the training's model folder,
  `<models folder>/model_<date>_<time>/`, and the "Training Started" message
  shows it.
- **Only one training from a window at a time.** Clicking Start Training again
  while one runs shows "Training Already Running" and starts nothing.
- **A training is finished only when it says so.** When it ends with a model
  file, the training writes `DONE.json` into its model folder (its tracker id
  and the model file). The window checks every 5 seconds; once the process has
  ended it reads that file. No `DONE.json` means the training did not finish:
  the window says "Training Did Not Finish" and nothing is classified -- even
  if other model files appeared in the models folder meanwhile (they are not
  this training's). A training that reports success but saved no model file is
  marked `failed` in the tracker and writes no `DONE.json`.
- **Auto-classify** (the "Auto-classify after training completes" box): when
  the training finished, the window selects THAT training's model in the model
  list and runs classification on the brain that was selected when the
  training started. It classifies nothing, and says why, when another brain is
  selected by then (select the brain and classify by hand) or when the model
  is not in the model list (click Refresh, pick it, classify by hand). An
  automatic classification loads its results and compares them with the
  previous run without asking; a classification you start yourself always asks.

## Step 3: Count by Region

```bash
python 6_count_regions.py --brain YOUR_BRAIN
```

Outputs `cell_counts_by_region.csv` in the brain's `6_Region_Analysis/` folder.

### Counting from Setup & Tuning ("10. Results" tab, "Run Region Counting")

Counting always counts the brain's **production** classification,
`5_Classified_Cells/cells.xml`. The line "Counting uses:" at the top of the
Results tab says which run that file came from (from `promoted_from.json`), or
that there is none yet. To count a different run, promote it first (see above).

Opening the "10. Results" tab (or clicking "Refresh Views") reloads its "Allen
Atlas" and "eLife Groups" tables from the selected brain's count files in
`6_Region_Analysis/`. Opening any other tab does not touch them.

"Run Region Counting" runs step 6 in the background with no time limit (a
whole brain takes far longer than the five minutes it used to be allowed). Meanwhile napari stays usable, the
status line shows step 6's latest message, the brain list is held, and a second
click says "Counting Running". When it ends the counts are loaded into the
Results tab's "Allen Atlas" and "eLife Groups" tables (only if the counted
brain is still the selected one), the status line says "Complete - N cells in
M regions", and the count is written to the session log. If the counts were
written but cannot be shown, the status line says so in red; the count itself
is still on disk and logged. The Results tab does not compare with the
published reference; use `util_compare_to_published.py` for that (see below). Closing the
Setup & Tuning window while counting runs is refused with a message: wait until
the status line says it finished. If napari itself is closed anyway (or
stops), the count can be cut off part-way and its tracker row left at
`started`: run Region Counting again for that brain (or run step 6 from the
command line).

## Step 4: Compare to Published Data

```bash
# Compare your counts to Wang et al. 2022 eLife reference
python util_compare_to_published.py --brain YOUR_BRAIN

# Just see the reference values
python util_compare_to_published.py --show-reference
```

### Reading the comparison output

```
Region                                        Yours      Ref     Diff   %Match
--------------------------------------------------------------------------------
Red Nucleus                                    1523     1860     -337    81.9%  OK
Gigantocellular reticular nucleus              6543     7997    -1454    81.8%  OK
Corticospinal                                  4521     8713    -4192    51.9%  !!
```

- `OK` = within 80-120% of reference
- `~` = within 50-150% of reference
- `!!` = outside expected range
- `MISS` = you have 0 but reference has cells

### Key regions to watch

These regions are most important for functional recovery analysis:
- **Red Nucleus** - major motor relay
- **Pedunculopontine nucleus (PPN)** - locomotion
- **Gigantocellular reticular nucleus** - motor output
- **Corticospinal** - voluntary movement

If these are consistently under reference, try a lower threshold or smaller
ball filter. If over, try the opposite, or check classification.

## Step 5: Review and Adjust

### If counts are too LOW (missing cells):
- Decrease `threshold`
- Decrease `ball_xy_size`
- Check if `soma_diameter` matches your cells

### If counts are too HIGH (false positives):
- Increase `threshold`
- Increase `ball_xy_size`
- Consider retraining the classifier with your false positives

### If specific regions are wrong:
- May indicate registration issues in that area
- Run `python util_registration_qc.py --brain YOUR_BRAIN` to check

## Tracking Your Progress

Every detection run is automatically logged. View your experiment history:

```bash
# See recent runs
python experiment_tracker.py

# See runs for a specific brain
python experiment_tracker.py --search YOUR_BRAIN

# See best-rated detection runs
python experiment_tracker.py --best detection

# See statistics
python experiment_tracker.py --stats
```

### Rating your runs

After each detection, you're prompted to rate it 1-5. Use this to remember which parameter combinations worked best.

## Example Optimization Session

```bash
# Start from the settled method (or, for a new imaging setup, all eight by hand)
python 4_detect_cells.py --brain 101_PROJ_01_02_2p5x_z5
python 5_classify_cells.py --brain 101_PROJ_01_02_2p5x_z5
python 6_count_regions.py --brain 101_PROJ_01_02_2p5x_z5
python util_compare_to_published.py --brain 101_PROJ_01_02_2p5x_z5

# Review output, then try a change
python 4_detect_cells.py --brain 101_PROJ_01_02_2p5x_z5 --threshold 9
# ... repeat classify, count, compare

# When a run is right: Setup & Tuning -> select it -> Mark as Paradigm Best
```

## Reference: Detection Settings

There are no built-in presets. Detection uses the settled method for the
brain's imaging setup, read from the tracker -- see "The Settled Detection
Method" in README.md for what it contains and how to change it.

## Reference: Published Baseline

From Wang et al. 2022 eLife (L1-injected uninjured, n=5):

| Region | Average | StdDev |
|--------|---------|--------|
| Corticospinal | 8713 | 433 |
| Gigantocellular reticular nucleus | 7997 | 1520 |
| Pontine Reticular Nuclei | 2185 | 941 |
| Red Nucleus | 1860 | 1050 |
| Pontine Central Gray Area | 1528 | 471 |
| Lateral Reticular Nuclei | 1441 | 568 |
| Hypothalamic Lateral Area | 1170 | 419 |
| Magnocellular Ret. Nuc. | 1097 | 528 |
| Solitariospinal Area | 878 | 252 |
| Midbrain Reticular Nuclei | 646 | 219 |
| Medullary Reticular Nuclei | 635 | 139 |
| Raphe Nuclei | 535 | 258 |

See `util_compare_to_published.py --show-reference` for the full list.
