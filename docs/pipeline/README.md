# BrainGlobe Pipeline Scripts

Complete automated pipeline for processing lightsheet microscopy data through BrainGlobe tools.

## Pipeline Script Order

```
1_organize_pipeline.py         → Set up folder structure, move IMS files

2_extract_and_analyze.py       → Extract TIFFs, auto-crop spinal cord
   └── util_manual_crop.py     → (Optional) Adjust crop manually in napari

3_register_to_atlas.py         → Register to Allen Mouse Brain Atlas
   ├── (auto) util_registration_qc.py  → Generates detailed QC images
   └── util_approve_registration.py    → (REQUIRED) Review & approve QC

4_detect_cells.py              → Detect cell candidates
5_classify_cells.py            → Classify cells with trained model
6_count_regions.py             → Count cells by brain region
```

**Important:** Step 4 will block until registration QC is approved -- so that you
don't spend hours of compute on a badly registered brain. The approval is for one
registration: registering again needs a new approval, and no automated run ever
approves (see "Registration QC & Approval" below). That gate applies to
the cropped images, which are the ones registration is computed from.

Step 4 can also be run **before** cropping and registration, on the whole
extracted stack, which is useful when registration is waiting on something. It
is a trial run: it gives you a cell count but not counts per brain region. See
[Running detection before registration](#running-detection-before-registration).

## Which step is a brain at?

`RUN_PIPELINE.py`, the napari pipeline dashboard (Plugins -> BrainTools -> 3D:
1. Pipeline Dashboard) and `overnight_batch.py` all show each brain at a step.
Nothing records that step; it is worked out every time from which files exist,
and all three use the same rules (`mousebrain.pipeline_status`), so they always
agree with each other and with what the next script will do. The brain's step
is the first item on this list that is not done yet:

| Step | Done when |
|---|---|
| Extract | `1_Extracted_Full/metadata.json` exists (extraction writes it last, so it means every channel finished) |
| Crop | `metadata.json` exists in `2_Cropped_For_Registration` or `2_Cropped_For_Registration_Manual`, or the brain was registered on its full extraction (it arrived already cut, so there is nothing to crop) |
| Register | a FINISHED registration is in `3_Registered_Atlas`: all four of `brainreg.json`, `registered_atlas.tiff`, `boundaries.tiff` and `deformation_field_0.tiff` (see "A registration that did not finish" below) -- and not rejected by a person, and not out of date (see "Stale outputs"). A registration that is running right now is not finished either: the brain stays at Register with "being registered right now", and nothing starts a second one |
| Approve | a person approved this registration (see "Registration QC & Approval" below) |
| Detect | a candidates file is where step 5 will look: the top of `4_Cell_Candidates`, or `4_Cell_Candidates/from_<the folder the atlas was fitted to>` |
| Pre-filter | optional (RUN_PIPELINE only). Done when any pre-filter output exists -- a `prefilter_report.json` in a `prefiltered_<time>` folder, or step 5's `prefiltered/interior_candidates.xml`. It is never shown once the brain is classified: step 5 pre-filters on its own by default |
| Classify | `5_Classified_Cells/cells.xml` exists |
| Count | a `.csv` exists in `6_Region_Analysis` |

**An extraction that was stopped part-way.** Running step 2 again on a brain
whose `1_Extracted_Full` already holds planes reuses them (fast) ONLY when they
are a finished extraction: every channel holds exactly `Z0000.tif` up to the
last plane with no gaps, as many planes as the `.ims` file says the brain has,
and each channel's last plane reads cleanly. Otherwise step 2 prints `[!] The
planes already in 1_Extracted_Full are not a finished extraction: <why>` and
extracts the brain again from the `.ims` (slow; nothing is lost). Each plane is
written under a hidden temporary name and renamed when whole, so a plane named
`Z####.tif` is never half-written. Planes beyond the brain's depth, left by an
earlier different extraction, are moved to
`1_Extracted_Full/_archive/<date>_<time>_extra_planes/`. (Before, any plane on
disk counted: an extraction stopped part-way came back as a "complete" brain
with its back planes missing, and every later step used the short stack.)

**Stale outputs.** A file that exists but was made from something replaced since
is shown as `[!] ... STALE: <why>` and its step is offered again (nothing is
deleted or moved). The registration is OUT OF DATE when the images it was
fitted to were made again after it (a re-crop) or are gone: every reader shows
`[!] Registered to atlas -- out of date: ...`, the brain is back at
"Register", its approval no longer counts (step 4 will not run, no Approve
button accepts it) and step 3's `--batch` registers it again. Counts are stale
when they are older than the current registration or the classification, or
were made on an out-of-date registration; a classification when it is older than its
candidates or the images; detection when it is older than the images the atlas
is fitted to (or, for a registration that does not record its images, older
than the registration). Anything made from a stale output is stale too.

A trial detection run kept in `from_1_Extracted_Full` on a brain that was
registered on its crop does NOT count as detection: those coordinates are not
in the space the atlas was fitted to, and step 5 will not read them.

**Which file is "the candidates".** Step 5 classifies one file: in the
registered-space candidates folder (the top of `4_Cell_Candidates`, then
`from_<the folder the atlas was fitted to>`), `detected_cells.xml` if it is
there, else `cell_classification.xml`, else the first `.xml` by name. Everything
else that reads "the candidates" reads that same file: `batch_process.py`'s
detection check, the "detected" mark in step 4's own list of brains
(`4_detect_cells.py` run without `--brain`), the cells drawn on the overnight
batch's sample pictures, the pre-filter (`python -m mousebrain.prefilter --brain ...`
without `--candidates`, and the Setup & Tuning pre-filter button when no run is
selected) and the Setup & Tuning Classify button. Nothing reads a run kept in
`4_Cell_Candidates/_archive/` (an earlier, replaced run) or a trial run outside
registered space. To use any other file, give it explicitly (`--candidates` on
the pre-filter, or choose a run in Setup & Tuning's pre-filter section with
**Load Best** or **Load Selected**).

**The Setup & Tuning pre-filter button and a chosen run.** Once a detection run
is chosen with Load Best or Load Selected (the label under the buttons shows
`Loaded: <run id>`), the button records that run as the pre-filter's parent --
so the run must be the one that wrote the file being filtered. The file filtered
is always the brain's own candidates file, the one step 5 classifies (in
`4_Cell_Candidates`). The chosen run's file is the file its `output_path` in the
tracker names, or `detected_cells.xml` / `cells.xml` in the folder it names; if
that is not the brain's own candidates file, nothing is pre-filtered.

- **A chosen run applies only to its own brain.** Loading another brain clears
  the choice (the label goes back to `No detection loaded`). If the run chosen
  is still somehow a run of a different brain, the button stops with "Run
  Belongs To Another Brain", naming both brains, and pre-filters nothing.
- **The file must be the run's own.** Step 4 writes a record beside each
  candidates file (`detected_cells.provenance.json`) naming the run that wrote
  it. If that record names a different run (a later run replaced the file), or
  the run's tracker row names only the `4_Cell_Candidates` folder and the file
  there has no record (old runs, and runs that never finished, recorded the
  folder; a newer run may have written the file since), the file is not taken.
- **A Setup & Tuning test run cannot be pre-filtered.** The button stops with
  "Run Cannot Be Pre-filtered", naming the run, and pre-filters nothing. Why: a
  test run keeps its cells in a folder of its own, outside the brain. A test of
  part of the stack covers only the planes tested, so its interior / removed
  counts would not be the brain's; and test runs saved before run files held
  whole-stack positions count each cell's plane from the first plane tested,
  so against the atlas every cell would sit in the wrong plane (see "Where a
  test run's cells are" in SYSTEM_ARCHITECTURE.md). A whole-stack test run is
  refused as well, because nothing records that the images it read are the
  ones the atlas was fitted to. What to do: run step 4 (`4_detect_cells.py`) with the settings
  you settled on, then choose that run and pre-filter. Or press **Clear Chosen
  Run** (beside Refresh) so no run is chosen, then **Run Pre-Filter**: the brain's
  candidates file is filtered with no parent run recorded. (Refresh, or reloading
  the same brain, keeps the chosen run; only Clear Chosen Run or loading a
  different brain forgets it.)
  A run the tracker lists as covering part of the stack (`det_scope` =
  `partial`) is refused wherever its file is.
- If no such file can be found (the run is not in the tracker, its file was
  moved or deleted, or the file is not shown to be its own), the button stops
  with "Selected Run Not Found", naming the run, and pre-filters nothing;
  choose another run.

It never filters a file under the name of a run that did not write it. When no
run has been chosen it filters the same file and records no parent run.

**Load Previous** (beside the list of earlier pre-filter runs) loads the
result of the run picked in that list, so it can be looked at, saved or sent
to training again. It loads only a run of the brain selected now; a run of
another brain stops with "Run Belongs To Another Brain" and loads nothing.
(Before this was corrected the button always ended in "Could not load previous
result" and loaded nothing.)

**The Setup & Tuning Classify button and a chosen run.** Classify always
classifies the file step 5 classifies, whatever run is chosen. The tracker row
it writes names, as the classification's parent, the detection run that wrote
**that file**:

- the chosen run (Load Best / Load Selected), but only if it is a run of the
  loaded brain and its own candidates file (found as described above) is the
  file being classified;
- otherwise the run named in the record beside the file
  (`detected_cells.provenance.json`);
- otherwise no parent at all. An empty parent means "not known", which is
  true; naming the chosen run for a file it did not write would not be.

A classification of a pre-filter output still names the pre-filter run as its
parent, as before.

## Which folders are brains?

Every script, the napari plugins, the utilities and the `mousebrain` launcher
(`mousebrain --crop <brain>`) -- with the one exception of the training-data
collectors described at the end of this section -- use the same rule
(`mousebrain.config.iter_brains`) to decide which folders are brains, so they
always agree on which brains exist. A brain is a folder two levels down,
`1_Brains/<mouse folder>/<brain folder>`, where:

- the **mouse folder** does not start with `_` or `.`, and its name (in any
  case) does not contain a skip word:
  - `script`, `backup`, `archive` or `summary` **anywhere** in the name, even
    glued to other letters: `Mouse_Backup`, `brainbackup`, `archivedmice`,
    `Archive2024`, `scripts` and `Summaries` are all passed over. This is how
    the steps have always matched these words, and it keeps a backup copy of a
    whole mouse from being processed as brains.
  - `old` only as a whole **word**: `old_deliveries`, `Old copies` and
    `mouseOld` are passed over, but `GOLD_01`, `HOLDOUT_A`, `Threshold_Test`
    and `scaffold-3` (which only contain the letters) are mice. The words of a
    name are split at `_`, `-`, spaces, dots, between letters and digits, and
    where lower case turns to upper case. One consequence: `oldbrains` (the
    word glued to other lower-case letters) IS a mouse folder -- name such a
    folder `old_brains`, or start it with `_`;
- the **brain folder** does not start with `_` or `.`.

**To keep something beside a brain without it being treated as one, start its
name with an underscore** -- for example `_superseded_<date>` for the outputs of
an earlier delivery. Such a folder is never listed, never processed and never
found by name -- not by its own name and not by its full
`<mouse folder>/<folder>` path either. The same holds for a brain inside a
backup or archive copy of a mouse: asking for it by `<backup folder>/<brain>`
prints why it is not a brain and does nothing.

**The exception: training data.** The places that gather labelled training
cubes for the classifier do not use this rule: Setup & Tuning's list of training
sources (each per-brain source is shown as `Brain: <folder name>`), its table of
saved training batches, and `scripts/utils/util_train_model.py`. They look in
`training_data` inside **every** folder two levels down, including folders that
start with `_` and mouse folders with a skip word. They gather labels a person
has already made, wherever they were saved; they do not decide which brains are
processed. So:

- a `_superseded_<date>` folder (or a backup copy of a mouse) that holds a
  `training_data` folder shows up as a training source under its own folder
  name. In Setup & Tuning, untick it in the training sources table if it should
  not be used (for example when it is a copy of cubes the live brain also has,
  which would count the same cubes twice);
- `util_train_model.py` has no such table: it uses every matching
  `training_data` it finds. To leave a copy out, move its `training_data`
  folder outside the brains root before training.

**When a brain is missing from a list.** The command-line listings (each step's
scan or list of brains when it is run without `--brain`, `RUN_PIPELINE.py`,
`overnight_batch.py`) print one
line for every folder in the brains root passed over because of a skip word:
`[i] Not treated as a mouse folder: <name> (its name has the word '<word>' ...)`.
Asking for a brain by name (`--brain`, or `mousebrain --crop <brain>`) that sits
in such a folder, or in a folder starting with `_`, prints why it was not found:
`[i] <brain> is in <folder>, which is not treated as a mouse folder (...)`.
These print that line: every step (2 to 6), the pre-filter, the orientation
check, `mousebrain --crop`, and `util_registration_qc.py`,
`util_manual_crop.py`, `util_approve_registration.py`,
`util_apply_crop_from_reference.py`, `util_compare_to_published.py`,
`util_pipeline_comparison.py`, `util_visualize_3d.py`, `util_optimize_crop.py`
and `util_create_training_cubes.py`. Two options only FILTER a list by part of
a name and print no such reason: `overnight_batch.py --brain` (a brain it
leaves out is still named with its own reason, see "The overnight batch") and
`util_generate_elife_report.py --brains` (it filters the brains already in the
counts file; a name that fits none simply gives an empty report). To bring
such brains back, rename the mouse folder. (The napari lists do not print
these lines; check the names there by hand.)

**When a folder cannot be read.** Two more lines are printed by EVERY listing
and every lookup by name -- the command-line ones and the napari lists alike
(for napari, in the terminal window it was started from):

- `[!] Could not read the brains folder <folder>: <error> ...` -- the brains
  folder itself is missing or cannot be opened. Nothing can be listed or found.
  The usual causes: the data drive is not connected, or the root folder is set
  to the wrong place (`mousebrain --paths` shows where it points). Steps 4, 5
  and 6 and `util_optimize_crop.py`, given `--brain`, stop with
  `ERROR: Could not read the brains folder ...` instead of calling the brain
  "not found"; `overnight_batch.py` writes the same line to its log, then
  `Nothing was processed.`, and ends as a failure instead of reporting
  "Found 0 brain(s)". Fix the drive or the setting and run again.
- `[!] Could not read the mouse folder <folder>: <error>. Its brains are missing
  from this list.` -- one mouse folder cannot be opened (usually its
  permissions). The other mice are listed as usual; the brains of that one are
  not listed, not found by name, and not processed, until the folder can be
  read. `overnight_batch.py` also writes this to its log as a warning
  (`Could not read <folder> ... were NOT listed and will not be processed by
  this run`).

A brain that silently drops out of a list looks exactly like a brain that is
not there, so these lines are never switched off.

What a step needs INSIDE a brain is still its own business: extraction lists
brains with a `.ims` in `0_Raw_IMS`, registration brains with images to register,
and so on.

The same rule decides what the napari brain lists offer (the Pipeline Dashboard,
Setup & Tuning, Manual Crop, the 3D cell viewer and the cross-brain comparison)
and which folder a utility acts on when you give it a brain by name
(the scripts named above). Before this, those lists offered an archive kept beside a brain
as if it were a brain, and a utility could act on it instead of the real one.

**Giving a brain by name.** The exact brain folder name always works. The
form `<mouse folder>/<brain folder>` (the way steps 3 to 6 print a brain) works
in steps 3 to 6, the pre-filter, `mousebrain --crop`, the orientation check and
the `util_*.py` scripts listed above as printing why a brain is not found;
write it with a forward slash (`/`). It does NOT work in two places, which
compare what you give with the brain folder name alone: step 2 (`2_extract_and_analyze.py --brain`), and
`overnight_batch.py --brain`. Give those the brain folder name without the
mouse folder. What a PART of a name does depends on the tool:

- `util_registration_qc.py`, `util_manual_crop.py`, `util_approve_registration.py`,
  `util_apply_crop_from_reference.py`, `util_compare_to_published.py`,
  `util_pipeline_comparison.py`, `util_visualize_3d.py` and the orientation
  check accept a part of a name only when it fits exactly one brain; when it
  fits several -- two imaging setups of one mouse share its number -- they stop
  and print the full names to choose from, instead of acting on whichever brain
  happened to be listed first.
- Steps 2, 4, 5 and 6, the pre-filter (`python -m mousebrain.prefilter --brain`),
  `util_optimize_crop.py` and `util_create_training_cubes.py` take the exact
  name only; a part of a name is "not found". Give the full folder name.
- Step 3 (`3_register_to_atlas.py --brain`) registers the exact brain when the
  name is exact (the brain folder name, or `<mouse folder>/<brain folder>`),
  and otherwise every brain whose folder name contains the part.
- `overnight_batch.py --brain` keeps every brain whose folder name contains
  what you give.

`mousebrain --crop <brain>` behaves like the first group: it opens only a brain (never a
`_` folder or a copy in a backup mouse folder), refuses a part of a name that
fits several brains and prints their full names, and when nothing fits it lists
the brains there are.

**The overnight batch** (`overnight_batch.py`) takes every brain that is already
registered (on whatever images) or has a crop to register -- a hand crop in
`2_Cropped_For_Registration_Manual` or an automatic one in
`2_Cropped_For_Registration`. Before it starts it lists every brain it leaves
out, with the reason:

- `not cropped and not registered yet` -- the brain has only its full
  extraction. Crop it (Plugins -> BrainTools -> 3D: 3. Manual Crop), or, if it
  arrived already cut, register it once on the full extraction with
  `3_register_to_atlas.py`; the next overnight run then takes it.
- `no extracted images yet` -- the brain has its `.ims` in `0_Raw_IMS` (or an
  empty `1_Extracted_Full`) but no images; run `2_extract_and_analyze.py` first.

A folder in a mouse folder that is not a brain at all -- a `training_data`
folder, a stray copy of a numbered output folder -- has none of `0_Raw_IMS`,
`1_Extracted_Full` or a crop or registration, and is not listed: the overnight
batch passes over it without a word, as `RUN_PIPELINE.py` does.

If the brains folder itself cannot be read, the batch logs
`Could not read the brains folder ...` and `Nothing was processed.` and ends as
a failure; if one mouse folder cannot be read, it logs a warning naming it and
carries on with the others (see "When a folder cannot be read" above).

(It used to take only brains with images in `2_Cropped_For_Registration`, so a
hand-cropped brain, or one registered uncropped, never ran overnight and nothing
said so.)

**`overnight_batch.py --force` registers the brain again.** `--force` redoes
every step, registration included: for a registered brain it runs
`3_register_to_atlas.py --batch --brain <brain> --reprocess`, which moves the
old registration and its approval into `3_Registered_Atlas/_archive/<date>_<time>/`
and registers again. The new registration has no approval, so the brain stops
there ("awaiting approval" in the report): detection, classification and
counting wait until a person has looked at the new QC pictures and approved
them; the next run then does them. If step 3 ends without a new registration
in the folder, the batch reports the registration as failed and does nothing
after it. Use `--force --brain <brain>`. `--force` without `--brain` is
REFUSED: it would register every registered brain again (hours each) and set
aside every approval, so the batch stops before touching anything, prints how
many brains that would have been, and ends with an error. If every brain really
should be registered again, add `--all-brains` (`--force --all-brains`).
(Before, `--force` never registered anything -- step 3 said
"already registered" -- yet the batch logged "Registration complete" and redid
every later step on the old registration.)

**The overnight batch never registers a REJECTED brain again** -- not even with
`--force` -- and never starts a second registration of a brain that is being
registered right now. Such a brain is counted under "Not started" in the summary
and listed in the report's **Not Started** section with the reason: for a
rejection, who rejected it, when, and what to change. See "Rejecting is
recorded".

To register ONE brain again by hand: `python 3_register_to_atlas.py --batch
--brain <brain folder name> --reprocess`. It refuses a name that fits more than
one brain (it lists them), and needs `--batch`. It also refuses a brain that is
being registered right now (see "A registration that did not finish").

## The Pipeline Dashboard's Run button

In napari (Plugins -> BrainTools -> 3D: 1. Pipeline Dashboard), select a brain
and press **Run Next Step**. Every step it starts acts on **the selected brain
only** and asks no questions:

| Next step | What the button runs |
|---|---|
| Extract | `2_extract_and_analyze.py --batch --brain <selected brain>` |
| Crop | opens nothing: the dashboard asks you to open the crop tool yourself (**Plugins -> BrainTools -> 3D: 3. Manual Crop**); save the crop there, then press **Refresh** on the dashboard |
| Register | `3_register_to_atlas.py --batch --brain <selected brain>`. If a person REJECTED the brain's registration it asks first, with three buttons: **Register again** (runs the same with `--reprocess`), **Approve anyway...** (overrules the rejection, after a second question), **Not yet** (nothing changes). If the brain is being registered right now it says so and starts nothing |
| Approve | shows the registration QC picture and asks "Does the registration look good?" with three buttons: **Approve**, **Reject** (recorded, see "Rejecting is recorded") and **Not yet**. Closing the question with Esc or the window's close button is **Not yet**: nothing is recorded. If this brain's registration status has not been shown yet when you press Approve, nothing is approved: the status is shown and a "Look first" message asks you to look at the registration and press Approve again |
| Detect / Classify / Count | steps 4 / 5 / 6 with `--brain <selected brain>` |

The same holds for **+ Add New Brain** when you answer Yes to starting extraction at the end. While the
step runs the dashboard shows nothing of its progress. When it ends the dashboard
shows "Done!" or "Error - check console". The script's output is printed to the
console napari was started from **only when the step fails**; after a step that
succeeds nothing of its output is shown anywhere, so to follow a long step as it
runs, run the same command from a terminal instead.

**The "Registration" button under BrainGlobe Tools** opens brainreg's own
registration widget with the brain's voxel sizes filled in -- for trying
settings, nothing more. It does NOT fill in an output folder or an orientation:
choose both yourself, and choose an output folder OUTSIDE the brain's folder.
A registration made with that widget is not the brain's pipeline registration
(nothing archives the old one or its approval, nothing records which images it
used, the tracker has no row of it). The brain's registration is made only by
**Run Next Step** (step 3). (It used to point the widget at the brain's
`3_Registered_Atlas` with a fixed orientation, so a run there replaced the
pipeline registration without any of that.)

(Before this, Extract started step 2 in its interactive form, which waits for
an answer the dashboard cannot give, and then extracted every brain waiting;
Register registered every brain waiting, not the selected one. Step 3 given an
exact brain folder name now registers that brain only, even when its name is
part of another brain's name.)

## Utility Scripts

```
experiment_tracker.py          → Core module: CSV-based experiment logging
util_experiments.py            → Browse/search/rate experiments interactively
util_optimize_crop.py          → Find optimal Y-crop via iterative testing
util_train_model.py            → Train custom classification models
util_manual_crop.py            → Manual crop tool (napari plugin launcher)
util_registration_qc.py        → Generate registration QC visualizations
util_approve_registration.py   → Approve registration after QC review
```

## What's New

| Script | Purpose |
|--------|---------|
| `4_detect_cells.py` | cellfinder detection with the settled method (see below) |
| `5_classify_cells.py` | cellfinder classification with trained model |
| `6_count_regions.py` | brainglobe-segmentation regional counting |
| `experiment_tracker.py` | Central CSV logging for all experiments |
| `util_experiments.py` | Interactive CLI for viewing/rating experiments |
| `util_optimize_crop.py` | Find optimal crop using registration quality metrics |
| `util_train_model.py` | Train custom cell classification networks |
| `util_manual_crop.py` | Napari plugin for manual crop adjustment |
| `util_registration_qc.py` | Generate detailed QC images comparing brain to atlas |
| `util_approve_registration.py` | Review & approve registration before cell detection |

## Batch processing (`batch_process.py`)

`batch_process.py` takes brains through the steps one at a time and stops a
brain at the first check it fails, recording why in
`2_Data_Summary/pipeline_status.csv`. For each brain it runs extraction
(`2_extract_and_analyze.py --batch --brain <brain>`, that brain only), then
checks that `1_Extracted_Full` holds `metadata.json` and `ch*/` TIFFs. It then
needs a crop -- `ch0/` TIFFs plus `metadata.json` in
`2_Cropped_For_Registration_Manual` or `2_Cropped_For_Registration`; with none,
the brain stops as "needs manual crop": crop it by hand and run it again. Then
registration (`3_register_to_atlas.py --batch --brain <brain>`), after which the
brain always stops at "needs registration approval" until a person approves it.
A registration that did not produce ALL four of its output files
(`3_Registered_Atlas/brainreg.json`, `registered_atlas.tiff`, `boundaries.tiff`
and `deformation_field_0.tiff`; see "A registration that did not finish") stops
the brain as FAILED at `register`, with the missing files named in
`qc_register`; look at the registration step's output for that brain, fix the
cause, and run it again with `--resume --retry-failed`. A registration a person
REJECTED is not registered again: the brain stops as REVIEW at `register`, and
`qc_register` says who rejected it, when, and what to change (see "Rejecting is
recorded"). A brain another run is registering right now stops as FAILED at
`register` with "being registered right now"; run it again later.
(Until this was fixed every brain failed at extraction: the batch passed steps 2
and 3 an option they do not accept, and the checks looked for files the steps do
not write.)

`--brain` on step 2 takes the exact folder name. Without it step 2 extracts every
brain that still needs it, as before.

**The status file is safe to share.** Every change to `pipeline_status.csv` is
made holding `pipeline_status.csv.lock` (beside it) and written to a temporary
file that is swapped in whole. So a crash or a full disk never leaves the file
cut short, and two batches running at once do not drop each other's rows.
(The one case no lock file can cover: a batch whose computer stopped dead for
more than two minutes in the middle of a write -- asleep, or cut off from the
shared drive -- and then carried on. Its lock was taken over meanwhile, so its
write and another batch's can overlap. If that happened, run
`python batch_process.py --status` and re-run any brain whose row looks wrong.)

- A batch that finds the lock taken waits and prints
  `Waiting for the lock on pipeline_status.csv (held by ...)` -- normally for
  well under a second, because each change holds the lock only for the moment
  of the write and lets it go at once.
- A lock stays behind only when a batch crashed (or was ended) in the middle of
  a write. The next change takes it over after two minutes and prints
  `Took over a lock left ... s ago by: ...`. Seeing that line when no batch
  crashed is a fault worth reporting. When several batches are waiting on the
  same old lock, they take it over one at a time: for a few thousandths of a
  second there is a second file, `pipeline_status.csv.lock.break`, held by the
  one doing it, and the others leave the lock alone meanwhile. That file stays
  behind only if a batch dies in that instant; the next batch to wait removes
  it after two minutes and says so (`Removed pipeline_status.csv.lock.break ...`).
- A batch that has waited five minutes stops with a message naming the lock
  file and who holds it: delete that file only when no batch is running. The
  five-minute limit holds even when an old lock cannot be removed (some program
  keeps it open, or the folder refuses it), so a batch never waits forever.
- If recording a brain's failure in the status file itself fails, the batch
  says so (`Could not record this failure in the status file`) and goes on
  with the next brain.

**Resuming.** Before each step starts, the brain's row in `pipeline_status.csv`
is saved as `PENDING` with the step in `current_step`. If the batch is stopped
part-way (window closed, restart, task ended), `python batch_process.py --resume`
finds those brains and takes them through the steps again, **skipping the steps
the brain had already finished** -- a step is skipped when its output is there
and current, by the same "done" and "stale" rules as "Which step is a brain
at?" above. What always runs again on a resume:

- **the step the batch was stopped in**, and every step after it. A step
  stopped part-way may have written some of its files and not the rest
  (counting writes the brain's own CSV before the summary `region_counts.csv`
  and the registry entry that the database pulls), so its files are never
  trusted. A brain stopped during counting runs counting again, not detection
  and classification;
- **a step whose output is stale**, by the rules in **Stale outputs** under
  "Which step is a brain at?" above (and anything made from a stale output is
  stale too). A re-crop, or a re-extraction when the brain was registered on
  the full extraction, makes the registration out of date (see Stale outputs).
  The batch registers it again, and it then waits for a new approval.

The skipped steps are listed in the brain's `skipped_steps` column, and the
batch prints `Already done (output there and current) -- not run again` for
each. The QC check after each step still runs, done or not.

**Only `--resume` skips anything.** A run started with `--input` runs every step
for every brain it is given, whatever is already there (extraction and
registration still pass over a brain that has their output, as they always
have). That is the way to process a brain again on purpose. The rules above
cannot see a change of method: candidates made with an earlier detection
method, or a classification made with an earlier model, are "current" as far
as the files go, so `--resume` keeps them. **To make a step run again**, either
run the brain with `--input <its .ims file>`, or run that step's script for
the brain by hand (for example `python 4_detect_cells.py --brain <brain>`) and
then each later step's script the same way. (`--resume` does not pick up a
brain whose row says it is complete, so it will not run those later steps for
you.)

`--resume` also picks up a brain that stopped at "needs registration approval"
once a person has approved its current registration; until then it only lists
it. Add `--retry-failed` to retry brains that failed: the step the brain failed
at, and every step after it, are run again even when their files are there (the
failed step's output is what failed). This holds whether the step reported a
failure or crashed with an error: a crash is recorded as a failure AT the step
the brain was in (`failed_step` in `batch_process.py --status`), so that step is
run again too. Each new attempt clears the previous attempt's failure fields.
Older versions recorded a crash as a failure with an empty `failed_step`, and a
status file can still hold such rows: for those, the step named in
`current_step` (the step the brain was in when it crashed) and every step after
it run again, and if `current_step` is empty too, every step runs again.

## Where the scripts live

The scripts come with the MouseBrain code; nothing is copied anywhere. The
numbered steps are in the repository's `scripts/pipeline/`, the `util_*.py`
scripts in `scripts/utils/`. The napari pipeline dashboard, the tuning
widget's buttons (registration QC, crop optimisation, pipeline comparison,
region counting) and `batch_process.py` find each script by name beside the
installed code, so they always run the version that is installed. To run one
by hand, run it from its folder, e.g.
`python scripts/utils/util_registration_qc.py --brain 001_CNT_01_01_1p625x_z4`.

(Before the reorganisation the scripts were copied into the pipeline's
`3D_Cleared/util_Scripts/`. That folder is data, is empty now, and nothing runs
from it; until this was corrected the dashboard and those buttons answered
"script not found", and registration finished without its QC picture.)

The experiment tracker creates its CSV at:
```
<CONNECTOME_ROOT>\Tissue\MouseBrain_Pipeline\3D_Cleared\2_Data_Summary\calibration_runs.csv
```

It starts a new CSV there only when the `2_Data_Summary` folder already
exists, and it never creates that folder. If you see
`refusing to create a new calibration tracker: its folder ... does not exist`,
the data drive is not connected (reconnect it and run again) or
`CONNECTOME_ROOT` points at the wrong place (`mousebrain --paths` shows what is
seen). On a brand-new installation, create that one folder by hand. WHY: when
the drive answered for the root but not yet for `Tissue\MouseBrain_Pipeline` (a
network share reconnecting), the tools used to fall back to the old
`3_Nuclei_Detection` layout and start a second, empty tracker there, which later
runs appended to while the real one sat unseen. The old layout is now used only
where its `3_Nuclei_Detection` folder really exists.

## Usage Examples

### Script 4: Detection
```bash
# Interactive mode
python 4_detect_cells.py

# The normal case: the settled method for this brain's imaging setup
python 4_detect_cells.py --brain <brain_id>

# A trial with one setting changed (logged as "custom")
python 4_detect_cells.py --brain <brain_id> --threshold 9

# On the whole extracted stack, before cropping or registration
python 4_detect_cells.py --brain <brain_id> --source full
```

Step 4 reads all eight detection settings from the run marked Paradigm Best
for this brain's imaging setup (magnification and Z-step, taken from the brain's
folder name), so nobody has to remember them or type them. If there is no such
run, or it does not record all eight, step 4 stops and says what to do -- see
"The Settled Detection Method" below.

**Earlier candidates are kept.** Step 4 writes `detected_cells.xml` (at the top of
`4_Cell_Candidates`, or in `from_<folder>` for a run on images that are not the
registered ones). When a run succeeds, the file that was there moves to
`_archive/<date>_<time>/` in the same folder first -- step 5 never looks there --
and beside the new file `detected_cells.provenance.json` names the tracker run
that wrote it and its settings. In the tracker, that run's `output_path` is the
file itself, and the previous run's `output_path` is updated to its file's place
in the archive. A run that fails leaves the previous file where it was.

### Script 5: Classification
```bash
# Interactive mode
python 5_classify_cells.py

# With specific model
python 5_classify_cells.py --brain 001_CNT_01_01_1p625x_z4 --model path/to/model.h5
```

**The pre-filter (on unless `--no-prefilter`).** Before classifying, step 5
removes candidates within 100 um of the brain's edge (surface glow, damage,
uncleared tissue -- labelled neurons are not at the tissue edge) and candidates
far outside the atlas. What is left is written to
`4_Cell_Candidates/prefiltered/interior_candidates.xml` and is what gets
classified. Candidates that land inside the brain outline but in no atlas region
are kept.

**If the pre-filter fails, step 5 stops.** It prints `ERROR: The pre-filter
failed: <why>`, records a failed pre-filter run in the tracker, classifies
nothing and leaves the production classification as it was. Fix the cause, or
run step 5 with `--no-prefilter` if you mean to classify the unfiltered
candidates. Why: step 5 used to print a warning and classify the unfiltered
candidates -- surface glow included -- into the production classification, and
nothing on the run said so.

**Classifying a pre-filter you tuned and saved.** The pre-filter run from
RUN_PIPELINE's "Pre-filter candidates" step, from `python -m mousebrain.prefilter`,
or tuned in Setup & Tuning (Save pre-filter results) is saved as
`4_Cell_Candidates/prefiltered_<date>_<time>/interior_candidates.xml`. To classify
one of those instead of step 5's own default pre-filter:

```bash
python 5_classify_cells.py --brain <brain_id> --candidates <brain folder>/4_Cell_Candidates/prefiltered_<date>_<time>/interior_candidates.xml
```

RUN_PIPELINE asks for you: when a brain has saved pre-filters, its "Classify
cells" step asks "Which candidates should be classified?" -- step 5's own
pre-filter, or one of the saved ones by folder name. (The pipeline dashboard and
the batches always use step 5's own pre-filter.) A saved pre-filter is
classified as it is, never pre-filtered again. Step 5 refuses it -- and
classifies nothing -- when it is not a pre-filter of the brain's current
candidates: a pre-filter of another file (an archived or trial detection run),
one made before the candidates were detected again (`... were made again after
this pre-filter`), one that does not say which candidates it filtered, or step
5's own `prefiltered/` folder. Pre-filter the candidates again and save, then
classify that. Why: step 5 used to ignore saved pre-filters entirely, so a tuned
pre-filter was never used, although the pre-filter itself told people to pass it
with `--candidates`.

To decide where a candidate is, the pre-filter uses BrainGlobe's own transform
from the detection images into the registration -- the same transform step 6
counts with. Since 2026-10-27; before that it used a shortcut that reordered the
image axes but did not reverse them, so on every brain it looked candidates up
millimetres from where they are (up-down on brains imaged `iar`, front-back as
well on `ipr`) and removed an essentially arbitrary set as "surface". Any
classification made before that date went through that wrong pre-filter.

**The previous classification is kept.** Step 5 writes the brain's production
classification, `5_Classified_Cells/cells.xml` and `rejected.xml` -- the files
step 6 counts. While it works (hours), the run writes into
`5_Classified_Cells/_in_progress/<date>_<time>/`, and the production files stay
exactly where they are -- so a run that fails, is stopped with Ctrl-C, is
killed, or is cut off by a restart, a power loss or a lost network drive leaves
the brain's classification unchanged. Only when the run has finished and saved
its files does step 5, in one short step, move the files already there (and
their record, below) into `5_Classified_Cells/_archive/<date>_<time>/` (kept,
never deleted) and move the new files in. What a failed run wrote goes into
`_archive/<date>_<time>_failed/`. A run that was killed outright leaves its
`_in_progress/<date>_<time>/` folder behind; that folder is never counted, and
the next step 5 run of the brain moves it into `_archive/<date>_<time>_stopped/`.
(Should a stop land in the split second of the swap itself and leave
`5_Classified_Cells` with no `cells.xml`, the previous files are in the newest
`_archive/<date>_<time>/` folder and the new ones in `_in_progress/`: move
either set back into `5_Classified_Cells`.) Beside the new files,
`promoted_from.json` names the tracker run that wrote them (`exp_id`), how
(`"how": "step 5"`, or `"promoted"` for a Setup & Tuning run made production
with **Promote to production**), when, and where the previous files went. In
the tracker, the run's `output_path` is the new `cells.xml`, and the run that
wrote the previous files has its `output_path` moved to their folder in the
archive -- the same as step 4 does with candidates. Why: step 5 used to write
over the previous classification with no copy, and every step 5 row named the
same folder, so no row could be matched to the file it had written. To go back
to an earlier classification, move its files from the archive folder back into
`5_Classified_Cells` (moving the current ones aside first) and count again.

### Script 6: Regional Counting
```bash
# Interactive mode
python 6_count_regions.py

# Process specific brain
python 6_count_regions.py --brain 001_CNT_01_01_1p625x_z4

# Process all pending
python 6_count_regions.py --all
```

**How the cells are placed in regions, and no quiet second method.** Step 6
places each cell in an atlas region with BrainGlobe's own transform through the
registration. If that cannot run (brainglobe-utils is not installed, a file of
the registration is missing, any error), the brain is **not counted**: step 6
prints the reason and `[FAILED] Not counted`, its tracker row says failed, and
the other brains in the batch are still counted. There is a second, less
accurate method -- looking each cell up in the registered atlas image directly
-- and step 6 uses it only when you ask for it:

```bash
python 6_count_regions.py --brain <brain_id> --allow-direct-lookup
```

Every count records which method placed its cells, in the `count_method` column
of `region_counts.csv` and `elife_counts.csv` and in the analysis registry's
method parameters: `brainglobe_transform` (the normal one) or
`direct_atlas_lookup`. Why: step 6 used to switch to the direct lookup by
itself, and the counts it wrote looked like any others, so nobody could tell
which brains had been counted the less accurate way. Rows written before this
change have an empty `count_method`; they cannot be told apart, so count those
brains again if it matters.

What step 6 writes, besides the per-brain files in `6_Region_Analysis`:

- `2_Data_Summary/region_counts.csv` and `elife_counts.csv` -- one row per brain,
  keyed by `brain` (`<mouse folder>/<brain folder>`). These are the files an
  integrator pulls. Each is written to a temporary file and swapped in, so a
  crash never leaves a cut-off file.
- When a brain is counted again, its previous row moves to
  `region_counts_archive.csv` / `elife_counts_archive.csv`, written in the
  archive's own column order; new columns extend the archive's header. (Before
  this was fixed, archived rows after the first were written in a different
  order from the header, so their values sat under the wrong column names. If
  step 6 finds an archive like that, it renames it to
  `region_counts_archive.malformed-<date>_<time>.csv`, keeps it untouched, prints
  a warning and starts a new archive.)
- **Out-of-date counts leave these tables at once.** A brain's counts are
  regions of one registration, counted from one classification. When either is
  replaced, the brain's row is taken out of `region_counts.csv` and
  `elife_counts.csv` straight away and moved to their archives, with
  `archived_at` and a `withdrawn_reason` column saying why; step 6's per-brain
  copy in `2_Data_Summary` (`<mouse>_<brain>_counts.csv`) is renamed
  `<mouse>_<brain>_counts.withdrawn-<date>_<time>.csv` beside it, so an
  integrator no longer imports it. This happens when step 3 moves a
  registration into `_archive` (registering the brain again), when step 5
  finishes a new classification, and when a Setup & Tuning run is promoted to
  production. Each prints `Counts of <brain> withdrawn from ...`. The brain's
  own `6_Region_Analysis` files stay where they are, and RUN_PIPELINE and the
  dashboard show the brain at "Count" again: run step 6 for it and its row comes
  back. Why: the rows used to stay until someone counted the brain again, and
  nothing in the tables told them apart from current counts. A rejected
  registration or a re-crop makes the counts stale in the status readers but
  does not yet take the row out of the tables; count such a brain again after
  registering it again.
- **One run at a time changes these tables.** Step 6, the rebuild
  `python regenerate_derived_counts.py` (which remakes `elife_counts.csv` and
  `hemisphere_laterality_analysis.csv` from `region_counts.csv`), and the
  withdrawal of a brain's counts (below) each hold a lock file,
  `region_counts.csv.lock` beside the tables, while they read and rewrite
  them. A second run started meanwhile prints `Waiting for the lock on
  region_counts.csv (held by ...)` and carries on when the first one is done.
  Why: two runs at once used to read the same old file, and the second write
  dropped the first run's row. The lock file exists only while a run holds it;
  one left by a run that crashed is taken over after two minutes. If a run
  says it waited five minutes and gave up, and nothing else is counting, delete
  `region_counts.csv.lock` and run it again. The rebuild also archives the old
  eLife rows in the archive's own column order and writes every table through
  a temporary file, like step 6.
- The `det_*` and `voxel_*` columns: the settings of the detection run behind
  the brain's `cells.xml`, found by following the tracker from the
  classification that wrote `cells.xml` -- the run named in
  `5_Classified_Cells/promoted_from.json` when there is one (step 5 and
  **Promote to production** write it; before that, counts of a promoted
  Setup & Tuning run could not be traced at all, because that run's row names
  its own run folder). The record is believed only while `cells.xml` is not
  newer than the record itself: if something wrote `cells.xml` without
  writing the record (an older installed copy of step 5, a file put back by
  hand), the record names a run that did not make the file, so step 6 instead
  takes the newest finished classification run of the brain that names
  `5_Classified_Cells` and started before `cells.xml` was written. From the
  classification the chain goes through the pre-filter if there was
  one, to the detection run whose output those candidates are. If the chain
  cannot be followed (runs from before the tracker recorded these links), the
  `det_*` columns stay empty and step 6 prints `[!] The detection run behind
  this cells.xml could not be traced`; the voxel size then comes from the
  images' `metadata.json`. The `subject`, `cohort` and `brain_id` columns come
  from the brain folder name.
- An entry in MouseBrain's analysis registry, `Registry/exports/3D_Region_Counts/`
  (`mousebrain.analysis_registry`): the counts CSV, the total, the detection
  settings, the detection and classification run ids, the model, and the source
  `cells.xml` and registration. A registry problem is printed and does not undo
  the count. These entries are not judged by the slice-analysis approved
  method, so `mousebrain-registry` does not list them as stale unless your
  `approved_method.json` names `3D_Region_Counts` -- once the manifest records
  that declaration. A manifest written by an earlier version gets it on the
  next count of any brain, or at once with `--no-approved-method`;
  `--refresh-approved` alone does not change it (see
  [ANALYSIS_REGISTRY.md](ANALYSIS_REGISTRY.md), "Not every analysis has one").

### Utility: Browse Experiments
```bash
# Interactive browser
python util_experiments.py

# Quick commands
python util_experiments.py recent 20
python util_experiments.py search "001"
python util_experiments.py best detection
python util_experiments.py stats
```

`search` finds every run whose brain name CONTAINS the text you type (so
`search "001"` lists all brains with 001 anywhere in the name) and shows the
50 newest. That is only the typed search. Wherever the tools ask for ONE
brain's runs -- the run list in Setup & Tuning, "has this detection been
classified", the Best lookups -- the brain is matched by its whole name
(`<brain>` or `<mouse>/<brain>`) and every run is returned, however many the
brain has. (Until this was fixed, a brain's run list also showed brains whose
name merely contained it, and only its 50 newest runs, so an older Best could
silently drop out of the list.)

### Utility: Manual Crop
```bash
# Launch napari with brain loaded and crop tool ready
python util_manual_crop.py --brain 001_CNT_01_01_1p625x_z4

# Or use the napari GUI:
# 1. Launch napari
# 2. Plugins → BrainTools → 3D: 3. Manual Crop
```

**Cropping a brain again.** The new crop is first written to
`2_Cropped_For_Registration.partial` beside the brain's other folders. Only
when every plane and its `metadata.json` are written is the previous crop moved
into `2_Cropped_For_Registration/_archive/<date>_<time>/` (kept, not deleted)
and the new one moved in. If the crop is stopped part-way, the previous crop is
untouched and the `.partial` folder is left behind; the next crop deletes it and
starts again. A registration made on the previous crop is then OUT OF DATE (see
"Which step is a brain at?"): register the brain again and approve the new
registration. (Before, a crop wrote over the old one plane by plane: a crop
stopped part-way left old and new planes mixed under the old `metadata.json`,
and the old registration and its approval stayed valid for detection.)

### The orientation check (drawn by registration, or on demand)

Registration has to be told how the stack lies: which way each of its three axes
runs, as three letters such as `iar` (the brain's `metadata.json`, key
`orientation`; `iar` when nothing says otherwise). Told wrongly, it still
registers, for over an hour, and every cell then lands in the wrong region while
the numbers look normal. A re-stitched delivery can come back lying differently
from the first stitch.

So **before it starts, registration draws `3_Registered_Atlas/QC_orientation_check.png`**:
BrainGlobe's own "Check orientation" (the button in its registration window), as
a picture. The stack is turned into the atlas's orientation as declared and
averaged three ways -- seen from the front, from above and from the side -- under
the same three views of the atlas, each drawn at its real size in millimetres.

**How to read it:** if the declared orientation is right, each picture of the
brain (bottom row) lies the same way as the atlas above it -- same way up, front at
the same end (olfactory bulbs towards the top of the views from above and from the
side). A brain cut short by the person who prepared it only looks shorter. A
left-right mirror is nearly invisible here, because a brain is nearly symmetric;
look for a known one-sided feature (an injection site, damage) if a mirror is in
question.

To draw it for any brain, or to try another orientation before re-registering:

```bash
mousebrain-orientation-check 001_CNT_01_01_1p625x_z4
mousebrain-orientation-check 001_CNT_01_01_1p625x_z4 --orientation sar
```

A trial orientation is written as `QC_orientation_check_<orientation>.png`, so the
declared one's picture is never overwritten. Each picture is entered in
MouseBrain's analysis registry (analysis `3D_Orientation_Check`; like the
region counts, not judged by the slice-analysis approved method once its
manifest records that -- see ANALYSIS_REGISTRY.md) and on the
registration's tracker row. It takes about two minutes (it reads every sixth
slice and pixel). To register with a different orientation, set `orientation` in
the brain's `metadata.json` and run registration again.

### Utility: Registration QC & Approval
```bash
# Generate QC images for existing registration
python util_registration_qc.py --brain 001_CNT_01_01_1p625x_z4

# Review and approve registration (REQUIRED before cell detection)
python util_approve_registration.py --brain 001_CNT_01_01_1p625x_z4

# Just view QC without approving
python util_approve_registration.py --brain 001_CNT_01_01_1p625x_z4 --view

# Check approval status
python util_approve_registration.py --brain 001_CNT_01_01_1p625x_z4 --status
```

**A registration that did not finish.** brainreg writes `brainreg.json` first,
before it registers anything, so that file alone only says a registration
STARTED. A brain counts as registered only when all four output files are there
(`brainreg.json`, `registered_atlas.tiff`, `boundaries.tiff`,
`deformation_field_0.tiff`). What you will see:

- **Registration fails** (brainreg stops with an error, or its files are not all
  there afterwards): step 3 prints `[FAIL]` and moves what the failed run wrote
  into `3_Registered_Atlas/_archive/<date>_<time>_failed/` -- kept, so whoever
  looks into the failure can see it. The brain is back at "Register".
  Only a run that got as far as starting brainreg is archived as `_failed`. If
  step 3 cannot move the PREVIOUS registration aside before it starts (a file
  in use, a full disk), nothing is registered, nothing is labelled `_failed`,
  and step 3 says that moving the previous registration failed part-way: what
  was moved is in the newest `_archive/` folder, the rest is still in
  `3_Registered_Atlas`. Put it back together before registering again.
- **Registration was stopped part-way** (the window was closed, the computer
  restarted): the leftovers stay in the folder. RUN_PIPELINE and the pipeline
  dashboard show `[!] Registered to atlas -- the registration did not finish
  ... (missing: ...)` and offer "Register" again; Setup & Tuning shows
  `NEEDS REGISTRATION`. Step 3 lists the brain under `[INCOMPLETE/FAILED]`, and
  `--batch` (also the dashboard's Run button and the overnight batch) registers
  it again, first moving the leftovers into
  `3_Registered_Atlas/_archive/<date>_<time>_incomplete/`.
- **Nothing unfinished can be approved**: every Approve button and
  `util_approve_registration.py` refuse, naming the missing files.
- **A registration that is RUNNING is not "did not finish".** While brainreg
  works, the folder holds `brainreg.json` and nothing else for a long time --
  exactly what a stopped run leaves. So step 3 writes
  `3_Registered_Atlas/.registration_running` (which computer, which process,
  when it started) before brainreg starts, and deletes it when the run ends,
  whether it succeeded, failed or was stopped with Ctrl-C. While that file is
  there and its run is still going, every reader shows `being registered right
  now (on <computer>, process <n>, started <time>)`; step 3 -- `--batch`,
  `--reprocess`, the interactive menu, the dashboard's Run button, the
  overnight batch and batch_process -- refuses to touch the brain (nothing is
  archived, no second brainreg starts) and says so. "Still going" means: on the
  same computer, step 3's process OR the brainreg it started is still running
  (the file names both: `process <n> (brainreg: process <m>)`); on another
  computer, the file is less than 24 hours old (another computer's processes
  cannot be seen). brainreg is named because stopping step 3 does not stop it:
  when the overnight batch or batch_process gives up on step 3 after its 2-hour
  limit, or the dashboard is closed, brainreg keeps writing into the folder,
  and a retry must wait for it rather than start a second brainreg on top.
  Pressing Ctrl-C in step 3's own window stops brainreg too.
  If both are gone (the computer restarted, brainreg ended after step 3 was
  stopped) the registration counts as did-not-finish -- step 3 never checked
  it, nor recorded which images it was fitted to: the brain shows "the registration did
  not finish: the run ... is no longer running" and the next registration
  replaces the file. If you KNOW a run on another computer has stopped and do
  not want to wait 24 hours, delete `.registration_running` by hand.

**An approval belongs to ONE registration.** Approving writes
`3_Registered_Atlas/.registration_approved`, and that file records which
registration was approved (the size and time of the registration's own output
files: `brainreg.json`, `registered_atlas.tiff`, `boundaries.tiff`,
`deformation_field_0.tiff`). Every place that asks "is this brain approved?" --
step 4, RUN_PIPELINE, the napari pipeline dashboard and Setup & Tuning, the
overnight batch, batch_process -- compares that record with what is in the
folder now. What that means for you:

- **Registering a brain again needs a new approval**, even if the old
  registration was approved. When registration runs, the previous approval is
  moved into `3_Registered_Atlas/_archive/` together with the registration it
  was for. The brain shows "Approve" again until you look at the new QC
  pictures and approve them.
- **Approvals written before this rule** (an empty file, or one holding only
  "Approved: <date>") do not say which registration they were for, so they do
  **not** count. Such a brain shows "Approve" with the explanation "the approval
  on file does not say which registration it was for". Look at its current QC
  pictures and approve it again -- one click or one command.
- **Automated runs never approve.** The overnight batch and batch_process stop a
  brain at "waiting for approval" after registering it; their reports list
  those brains. Only a person approves: `util_approve_registration.py`, the
  Approve button in Setup & Tuning (Registration QC tab), the Approve button in
  the pipeline dashboard, or "Yes, approve it" in RUN_PIPELINE.
- **You cannot approve before registering.** With no `brainreg.json` there is
  nothing to approve, and the buttons say so.
- **Approve in Setup & Tuning signs the registration you were shown.** The
  status line on the Registration tab is read again every time you open that
  tab, when you select a brain, after Approve or Reject, and when you load a
  brain in Registration QC mode. Approve signs only the registration that line
  (and, if you loaded them, the QC pictures in the viewer) showed you. If the
  brain was registered again since then -- for example by a batch that
  finished while you were looking -- nothing is approved: a warning
  "Registered again since you looked" appears and the status line now shows
  the new registration. Look at the new one (load the brain again in
  Registration QC mode) and press Approve again.
- **Rejecting is recorded.** Saying the registration is NOT good -- the Reject
  button in Setup & Tuning, the **Reject** button in the pipeline dashboard's
  "Does the registration look good?", "No, reject it (recorded)" in
  RUN_PIPELINE, or "no" (or "n") in `util_approve_registration.py` -- writes
  `3_Registered_Atlas/.registration_rejected` (who, when, which registration).
  Only an explicit Reject / no does this: closing the dashboard's question
  (Esc, the close button) is "Not yet", and `util_approve_registration.py`
  asks again on anything that is not yes, y, no or n (a bare Enter included).
  Any approval goes into `_archive/` (kept, not deleted). From then on every
  reader shows `[!] Registered to atlas -- rejected by <who> at <when> --
  change what made it bad, then register it again ...` and the brain is at
  "Register"; region counts made on it are shown as STALE. No Approve button
  accepts a rejected registration.
- **Nothing unattended registers a rejected brain again.** Step 3 `--batch`
  (without `--reprocess`), `overnight_batch.py` (also with `--force`) and
  `batch_process.py` skip it and print who rejected it, when, and what to
  change: on the same images and settings brainreg would only spend hours
  making the same fit. What to do:
  - **Re-crop it.** A crop made after the rejection makes the registration
    "out of date", and from then on every batch registers it again by itself.
  - **Or change its orientation or voxel sizes** (in the crop's
    `metadata.json`) and ask for the registration yourself: the dashboard's
    Run Next Step offers **Register again**, RUN_PIPELINE's "Register" step
    asks "Register it again now?", or run
    `python 3_register_to_atlas.py --batch --brain <brain> --reprocess`
    (step 3's interactive menu lists it under `[INCOMPLETE/FAILED]` too).
  Registering again moves the rejection into `_archive/<date>_<time>_rejected/`
  together with the registration it was for.
- **Rejected by mistake?** In the pipeline dashboard select the brain, press
  Run Next Step, choose **Approve anyway...** and confirm. Or run
  `util_approve_registration.py --brain <brain>`: it says who rejected it and
  asks "Approve it anyway, overruling the rejection?". Either way the approval
  is written (its note says it overruled a rejection) and the rejection moves
  into `_archive/`. No other path overrules a rejection.

**Re-registering also makes the region counts stale.** The status readers
compare each output with what it was made from (see "Which step is a brain
at?"): counts older than the current registration are shown as `[!] ... STALE`
and the count step is offered again. Detection and classification are positions
in the images, so they stay valid when the atlas is re-fitted to the SAME
images; if the images were made again (a new crop), they are stale too.

**How to read `QC_registration_detailed.png`:** each row is a coronal slice
through the registered brain, front of the brain at the top row, back at the
bottom. Left, the brain; middle, the atlas regions registration placed on it;
right, the brain with those regions' outlines drawn over it. A good
registration has the outlines following the brain's edges and its visible
landmarks (the ventricles, the midline, the corpus callosum). The rows are
spread over the stretch the atlas covers. Tissue the atlas does not have --
brainstem beyond where the atlas stops, partway down the medulla -- has no
outlines and gets no region; tissue the brain lost (a torn olfactory bulb)
leaves outlines with nothing under them. Neither is a registration fault.

### Utility: Optimize Crop
```bash
# Test 0%, 10%, 20%, 30%, 40%, 50% crops
python util_optimize_crop.py --brain 001_CNT_01_01_1p625x_z4

# Quick mode (0%, 25%, 50%)
python util_optimize_crop.py --brain 001_CNT_01_01_1p625x_z4 --quick
```

## Running detection before registration

Normally detection is step 4 of 6, and it runs on the cropped images that step 3
registered to the atlas. Sometimes you want to detect cells before any of that
has happened -- most often because registration is waiting on something: a
better scan of the same brain, a crop you have not made by hand yet, or your own
review of the QC image.

You can. Cell detection never reads the atlas. Cropping and registration exist
to put cells into *atlas* space, and that is only needed when you want counts
per brain region, which is step 6.

**How to do it**

In the terminal:

```bash
python 4_detect_cells.py --brain <brain_id> --source full
```

In the GUI: launch `mousebrain`, open `Plugins -> BrainTools -> 3D: 2. Setup &
Tuning`, pick the brain and press Load. When a brain has no crop, the plugin
loads the uncropped stack automatically -- there is nothing extra to click.
Then tune and run detection as usual.

### Seeing the candidates again afterwards

**Loading the brain shows the images, not the candidates.** To put a finished
whole-stack run back on screen, load it from the **Context Panel's run list**
(the runs table for the current brain) rather than expecting it to appear with
the images.

This is a consequence of where the results are kept, and it is deliberate. The
plugin's "look in `4_Cell_Candidates` and load what is there" paths all read the
top level of that folder only, and a whole-stack run is written one level down,
in `from_1_Extracted_Full/`. If those scans reached into it, the classification
step could pick up coordinates from a space the atlas was never fitted to and
produce a confident, wrong answer. The run list works instead because it goes by
the output path the tracker recorded for that run, which points straight at the
subfolder.

**What `--source` means**

| `--source` | Reads from | Can produce region counts? |
|---|---|---|
| `auto` (default) | best available: manual crop, else automatic crop, else the whole stack | only if it landed on a crop |
| `manual` | `2_Cropped_For_Registration_Manual` | yes |
| `cropped` | `2_Cropped_For_Registration` | yes |
| `full` | `1_Extracted_Full` | only if the atlas was fitted to the whole stack (see below) |

A folder only counts as available if it actually contains `ch0/*.tif`. The crop
folders are created empty when the brain is first organised, and an empty folder
is not data.

**What you get, and what you don't**

You get a real cell count for those settings on that brain, candidate
coordinates you can look at on the images in napari, and a logged run in the
calibration tracker like any other.

You do not get counts per brain region, and you cannot carry these cells forward
into steps 5 and 6. The reason is coordinates: the atlas is fitted to a
*cropped* stack, so its coordinates start from a different corner than the whole
stack's do. Cells found on the whole stack do not line up with an atlas fitted
later. When the brain is finally cropped and registered, **detection is run
again** on the crop.

So treat the number as provisional. It answers "do these settings find cells
here, and roughly how many" -- which is exactly what you want to know while
registration waits.

**The exception: a brain registered on the whole stack.** A brain delivered
already cut needs no crop, so registration is run on `1_Extracted_Full` itself.
Registration writes down which images it used (`3_Registered_Atlas/registered_from.json`),
and from then on the whole stack IS the atlas's space: steps 5 and 6 read their
images from `1_Extracted_Full`, and step 5 picks up a run in
`from_1_Extracted_Full/` (after anything at the top level). Nothing needs moving
or re-running. A brain whose registration has no `registered_from.json`
(registered before it was written) is treated as registered on the crop.

`registered_from.json` is written only when registration FINISHED (all its
output files are there); a failed run writes none. If the folder it names no
longer holds any images (moved, deleted, emptied), nothing reads another folder
in its place, because cell positions only mean something in the images the
atlas was fitted to. The message is "the registration of <brain> was fitted to
the images in <folder>, and that folder holds no images". What each tool does:
step 5, the pre-filter and Setup & Tuning stop with it. Step 6 does NOT count
that brain -- it prints `[FAILED] Not counted:` with the message, writes no
counts for it (it does not fall back to the less accurate direct atlas
lookup either), and goes on with the other brains; its summary line counts
the brain as failed. Put the images back, or register the brain again on the
images that are there now.

**Where the results go**

Into a subfolder named after the images they came from:

```
4_Cell_Candidates/
├── from_1_Extracted_Full/
│   └── detected_cells.xml      ← a trial run on the whole stack
└── detected_cells.xml          ← the real run, on the registered crop
```

Steps 5 and 6 look at the top level, plus `from_<folder>` only when the atlas
was fitted to that folder's images, so a trial run cannot be classified or
counted by mistake. The tracker records which images each run used
in its notes, for the same reason: two runs from different spaces are not
comparable as counts, and nobody should have to decode a path to notice that.

## The Settled Detection Method

Detection always uses **the settled method**: the detection settings someone
tuned on a real brain and then marked as the ones to use for every brain imaged
the same way. There is no menu of presets to choose from.

**Where it comes from.** In napari, `Plugins -> BrainTools -> 3D: 2. Setup &
Tuning`, a person tunes detection on a brain until the cells look right, selects
that run and clicks **Mark as Paradigm Best**. That run, in the calibration
tracker (`2_Data_Summary/calibration_runs.csv`), is the settled method for its
**imaging setup** -- the magnification and z-step at the end of the brain name
(for example `1p625x_z4`). Marking a different run later changes the method for
every brain detected after that; no code changes.

**What it contains.** All eight cellfinder settings, each its own tracker column:

| Setting | Tracker column |
|---------|----------------|
| ball filter XY size | `det_ball_xy` |
| ball filter Z size | `det_ball_z` |
| soma diameter | `det_soma_diameter` |
| threshold (SDs above mean) | `det_threshold` |
| ball overlap fraction | `det_ball_overlap` |
| soma spread factor | `det_soma_spread` |
| LoG sigma size | `det_log_sigma` |
| max cluster size | `det_max_cluster` |

All eight matter. Changing only the last four, with the first four held fixed,
moved most of the cells a run found (see the `det_ball_overlap` comment in
`src/mousebrain/tracker.py`). Runs logged before the last four columns existed
recorded only the first four; such a run cannot be used as the settled method
until its other four are filled in.

**Who uses it.** Step 4, the Pipeline Dashboard (`3D: 1. Pipeline Dashboard`:
the "Detection settings" box shows "Settled method (from tracker)" with the
selected brain's eight numbers and the run they came from), Setup & Tuning's
**Settled method (from tracker)** button, `batch_process.py`,
`overnight_batch.py` and `RUN_PIPELINE.py`.

**When there is none.** If no run is marked Paradigm Best for a brain's imaging
setup, or that run does not record all eight settings, detection does not run:
the dashboard shows the reason in red and step 4 stops with a message saying
what to do. It never falls back to built-in numbers. (Four generic presets --
sensitive, balanced, conservative, large_cells -- used to fill that gap. They
were never tuned on real brains, and the counts they produced looked plausible
and were not, so they were removed.)

A run marked only **Best** (for its one brain) is never taken in place of a
missing Paradigm Best. It used to be: with no Paradigm Best -- never marked, or
archived by mistake -- the newest brain-level Best of the imaging setup quietly
became the detection method of every brain imaged that way, and step 4 logged it
as the settled method. Now step 4 stops with the message above until someone
marks a Paradigm Best. The same rule applies to the classification model step 5
takes from the tracker: only a classification run marked Paradigm Best for the
imaging setup counts. Until a classification run is marked Paradigm Best (an
open decision), step 5 does not stop: it falls back to the newest model file,
exactly as before. (Setup & Tuning still shows a brain's Best; it just never
runs it for other brains.)

**Changing one setting for a trial.** Give it to step 4 on the command line,
e.g. `--threshold 9`. The run is logged as `custom`, never as the settled method.

**How the Best marks behave.** Each brain has at most one run marked Best (per
run type), and each imaging setup at most one Paradigm Best:

- *Mark as Best* moves the brain's Best mark to the selected run; other brains'
  marks are not touched. All three "make this the Best" buttons in Setup &
  Tuning do exactly this and nothing else: **Mark as Best** in the run list,
  **Mark Selected Layer as Best** and **Promote Recent to Best** in the
  detection comparison. The run that was Best only loses its Best mark: it is
  NOT archived, and it keeps its Paradigm Best if it had one. (The comparison
  buttons used to archive it, which also ended its Paradigm Best -- the
  settled method of every brain imaged the same way.) If you want a run
  archived, that is a separate decision.
- **The comparison buttons act only on the selected brain's layers.** Every
  points layer that shows a detection run remembers which brain and which run
  it is, and whether it was loaded as the Best or as the run to compare
  ("recent"). The buttons find the Best and recent layers by that, never by
  their names, and ignore layers of any other brain still open in the viewer.
  What you will see:
  - *Mark Selected Layer as Best* on a layer that is not a tracked run (one
    you made by hand, or loaded from a file) says "Not a Tracked Run" and
    changes nothing. On a layer of another brain it says so and changes
    nothing: select that brain first.
  - *Promote Recent to Best* and *Keep Current Best* need exactly one recent
    layer of the brain. With none they say so; with several (for example two
    detections run this session) they list them and ask you to close all but
    one, or to select the one you mean and use *Mark Selected Layer as Best*.
    If the recent layer shows the run that is already Best they change
    nothing.
  - After a mark, the new Best layer is renamed "BEST: ..." and the layer that
    showed the old Best is grayed out and renamed "[not best] ...". This is the
    viewer only; the tracker's Best mark has already moved.
  - *Keep Current Best* only grays out the recent layer in the viewer
    ("[archived] ..."); nothing is written to the tracker.
- *Mark as Paradigm Best* makes the run the imaging setup's settled method AND
  its brain's Best; the brain's previous Best is cleared (a brain never has two),
  and the previous Paradigm Best of that imaging setup loses that mark.
- *Archiving* a run clears both its Best and its Paradigm Best (no Best
  button archives a run any more; see above). An archived run is never used as
  the settled method, even if an old tracker row still carries the mark.
- "The brain" means the brain whichever way a run names it. A run made in
  Setup & Tuning is logged under the brain's own name (`<brain>`), a run made
  by a pipeline step under `<mouse>/<brain>`; both are the same brain. Marking
  either kind Best (or Paradigm Best) clears the Best of the other kind too.
  (It used to clear only a Best logged under exactly the same text, so a brain
  could end up with two Bests -- two stars in the run list -- and *Load best
  detection* loaded the older one. Marking the run you want Best again puts
  that right.)
- Each change is one locked write of the tracker, so a mark is never left half
  moved.
- *Load best detection* (the check box when loading a brain in Setup & Tuning)
  loads the run MARKED Best for the brain. If none is marked, it loads the
  highest-rated run that has not been archived and labels it "highest rated --
  none is marked best"; if there is none of those either, it loads nothing.
  A run made in Setup & Tuning records the candidates file it wrote
  (`detected_cells.xml` in its run folder); older widget runs recorded only
  the folder, and the file inside it is found and loaded. If that file is
  missing, the terminal says "Run ... has no candidates file at ..." and
  nothing is loaded.

## Folder Structure (Updated)

```
1_Brains/
└── 001_CNT_01_01/
    └── 001_CNT_01_01_1p625x_z4/
        ├── 0_Raw_IMS/
        ├── 1_Extracted_Full/
        │   └── QC_area_profile.png       ← Auto-crop detection visualization
        ├── 2_Cropped_For_Registration/
        ├── 3_Registered_Atlas/
        │   ├── QC_orientation_check.png      ← Was the orientation declared right? (drawn before registering)
        │   ├── QC_registration_detailed.png  ← Registration QC (auto-generated)
        │   ├── .registration_approved        ← A person's approval of THIS registration
        │   └── _archive/<time>/              ← Earlier registrations, each with its approval
        ├── 4_Cell_Candidates/            ← 4_detect_cells.py output
        │   ├── detected_cells.provenance.json ← which run wrote detected_cells.xml
        │   ├── _archive/<time>/          ← earlier detected_cells.xml, each with its record
        │   └── from_1_Extracted_Full/    ← trial runs made before registration
        ├── 5_Classified_Cells/           ← 5_classify_cells.py output
        ├── 6_Region_Analysis/            ← 6_count_regions.py output
        └── _crop_optimization/           ← util_optimize_crop.py output
```

## Experiment Tracking

All runs are logged to `experiments.csv` with:
- Unique experiment ID (e.g., `det_20241218_abc123`)
- All parameters used
- Timing information
- Results (cell counts, etc.)
- User ratings (1-5 stars)
- Notes

This eliminates the months-long manual optimization cycles by keeping a complete record of what was tried and what worked.

**How the tracker file is written** (`2_Data_Summary/calibration_runs.csv`). Several
machines write it, and every status change rewrites the whole file, so each write:

- holds a lock file beside it, `calibration_runs.csv.lock`, for the moment of the
  write (usually well under a second). Another machine that wants to write waits
  and prints `[Tracker] Waiting for the lock on calibration_runs.csv (held by ...)`.
  While a write is going on, the program doing it refreshes its lock every 30
  seconds, so a write that is slow because the share is slow is never mistaken
  for a crashed one. A lock nobody has refreshed for 2 minutes (by the file
  server's clock) was left by a process that crashed, and the next writer takes
  it over and prints `[Tracker] [!] Took over a lock left ... by: ...`. If a write
  ever reports that the tracker has been locked for 5 minutes and nothing is
  running, delete the `.lock` file. (This is the same lock the slice tracker and
  the analysis registry use, `mousebrain.file_lock`; a `.lock.break` file beside
  it is part of the takeover and is also safe to delete when nothing is running.)
- keeps the file's own column order and every column in it, including columns
  added by a newer version of the code -- an older copy of the code can no
  longer delete them.
- writes a temporary file and swaps it in, so a crash part-way leaves the
  previous file, never a cut-off one.

**A run is never left at `started`.** Steps 3-6 and model training
(`util_train_model.py`) mark their tracker row `failed` (with the reason in
`notes`) when they stop part-way -- an error, Ctrl-C, a crash inside brainreg or
cellfinder; a training that reports success but saved no model file is marked
`failed` too, and only a training that finished with a model writes `DONE.json`
into its model folder (what Setup & Tuning waits for). Setup & Tuning does the
same for its own detection and
classification runs when napari (or the Setup & Tuning panel) is closed while
one is still going: the row becomes `failed` with "napari was closed before the
run finished" in `notes`. (If only the panel was closed and the run then
finishes after all, its result is still recorded and the row becomes
`completed`.) Only a process killed outright (the task ended from Task Manager,
a power cut) can still leave `started`, because it runs no code at all on the
way out.

**Finding runs that never finished.** In a terminal, in the scripts/utils
folder:

```bash
python util_experiments.py unfinished        # every run still 'started'
python util_experiments.py unfinished 24     # only those begun more than 24 hours ago
python util_experiments.py view <ID>         # which machine ran it (Machine)
```

(or choose `8` in the interactive `python util_experiments.py` menu). Each run
listed is either still running on the machine shown, or was killed and will
never finish -- check that machine. Before this list existed, such rows could
only be found by opening the CSV.

## Requirements

```bash
conda activate MouseBrain    # or the full path of the environment if it was created with --prefix
pip install cellfinder brainreg brainglobe-segmentation brainglobe-atlasapi
pip install imaris-ims-file-reader tifffile numpy scipy matplotlib h5py
```
