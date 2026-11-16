# Counting cells in hand-drawn regions (brainslice-roi)

`brainslice-roi` lets a person draw regions (ROIs) on each slice image, then
counts the detected cells inside them. It works on the results of the 2D
detection batch (`python -m mousebrain.plugin_2d.sliceatlas.batch.batch_encr`),
so run that batch first.

Everything below is typed in a terminal where the environment MouseBrain is
installed in is active (for example `conda activate <your MouseBrain env>`).
`mousebrain roi ...` is the same command as `brainslice-roi ...`.

## The usual run

```
brainslice-roi batch <dataset folder> --region <REGION>
```

`<dataset folder>` is the folder holding the project's slice images (ND2
files, in any sub-folder). The command:

1. finds the detection results (see "Where it looks" below);
2. prints the dataset summary: how many samples already have ROIs, how many
   still need them, which ROI files cannot be read, and which samples are
   left out and why;
3. opens each sample in a viewer with two buttons, **Draw Left** and **Draw
   Right**. Draw, then click **Save & Next** (or close the window);
4. after the last image, counts the cells in every sample that has ROIs and
   writes the counts (see "What it writes").

Useful options:

| Option | What it does |
|---|---|
| `--region <REGION>` | Only samples whose name ends in that region code. |
| `--skip-done` | Open only the samples that still need ROIs (and those whose ROI file cannot be read). |
| `--count-only` | Do not open anything; just count the samples that have ROIs. |
| `--project <CODE>` | The project code the detection batch used. Default: the dataset folder's name. Give it when the folder is not named after the project. |
| `--detection-dir <folder>` | Read the detection results from exactly this folder (one sub-folder per sample, each holding `measurements.csv`). Use it when the batch was run with `--output <folder>`. |
| `-o <file>` | Where to write the combined summary CSV. |

## Where it looks for the detection results

In this order, and it uses the first folder that exists:

1. `--detection-dir`, if you gave it (nothing else is searched then);
2. `<registry>/exports/<project>_Detection/` -- where the detection batch
   records each sample's measurements (the batch prints this name on its
   `Registry:` line);
3. the detection batch's default output folder,
   `<2D data root>/2_Data_Summary/batch_results/`;
4. `<dataset folder>/batch_results/` (where older batches wrote).

If none exists, the command stops and prints every folder it tried. Then
either run the detection batch, or pass `--project` / `--detection-dir`.

**The fallback folder (3) is shared by every project.** The detection batch
writes every project's samples there unless it was given `--output`. When the
command falls back to it (because `<project>_Detection` does not exist in the
registry), the other projects' samples are in it too: their images are not in
your dataset folder, so each one is listed as left out with
`no <sample>.nd2 under <dataset folder>`. That list is not a fault in your
data. To read only your project's results, give `--project <CODE>` (so the
registry folder of step 2 is found) or `--detection-dir <folder>`.

## Samples left out

A sample in the detection results is **left out** -- not drawn, not counted --
when it has no `measurements.csv`, when no `<sample>.nd2` is found under the
dataset folder, or when its ROI file cannot be read. Every left-out sample is
printed with the reason, before the totals and again in the final
`SUMMARY: N samples counted, M left out` line. A sample outside the
`--region` you asked for is simply not part of the run and is not listed.

Two kinds of folder in the detection results are not samples and are never
listed:

- folders whose name starts with `_` -- the registry's own bookkeeping, for
  example `_archived`, where invalidated results are moved;
- folders holding ROI counts (`<sample>_roi_counts.csv`, directly or one
  folder down) and no `measurements.csv` -- this command's own output. Older
  versions wrote the counts into `<dataset folder>/batch_results/`, which is
  also searched for detection results.

## The ROI files, and what is never deleted

Each image's ROIs are saved beside it as `<image name>.rois.json`.

- **Clearing ROIs.** If you clear a sample's ROIs and close the window with
  nothing drawn, the old file is NOT deleted: it is renamed to
  `<image name>.rois.<date_time>.cleared.bak`, and the new name is printed.
  The sample then counts as having no ROIs. To undo, rename the `.bak` file
  back to `<image name>.rois.json`.
- **Clearing does NOT withdraw counts made earlier.** Neither does an ROI file
  becoming unreadable. Counts registered for that sample before stay as they
  were: its `<sample>__roi_counts` entry in the registry stays current, and
  its per-sample CSV and its row in `roi_summary_<REGION>.csv` stay in place.
  To withdraw them, invalidate that sample's entry (this moves its files into
  `_archived/<time>/` and marks the entry invalidated; see
  [ANALYSIS_REGISTRY.md](ANALYSIS_REGISTRY.md), "Invalidating results"):

  ```
  python -c "from mousebrain.analysis_registry import AnalysisRegistry; AnalysisRegistry(analysis_name='<project>_ROI_Analysis').invalidate(sample='<sample>')"
  ```

  Check the region summary CSV afterwards: the sample's row in
  `roi_summary_<REGION>.csv` is written by counting, and invalidating the
  entry does not remove it.
- **An ROI file that cannot be read** (damaged, half-written, not ROI data) is
  listed under "ROI file unreadable" with the error, and is not counted.
  `--skip-done` opens it so you can draw it again. If you draw new ROIs, the
  unreadable file is first renamed to
  `<image name>.rois.<date_time>.unreadable.bak`; if you close without
  drawing, it is left exactly as it was.
- The single-image `brainslice-roi draw` command follows the same rules.
- **Counting needs the image size.** A file this command saves records it
  (`image_shape`). A file saved from the napari widget's **Save ROIs** button
  does not. Its ROIs are still kept and shown: `brainslice-roi view` shows
  them, and `brainslice-roi batch` lists the sample under "ROI file has no
  image size: open it once in brainslice-roi batch and click Save & Next" --
  not under "unreadable", and not counted yet. To fix it, open that sample in
  `brainslice-roi batch` (`--skip-done` opens it too): the editor opens with
  its ROIs already loaded. Click **Save & Next** without changing anything.
  The ROIs are saved again with the image size, and the sample is counted.
  The old file is kept beside it as
  `<image name>.rois.<date_time>.noshape.bak`. `brainslice-roi draw` on that
  image does the same when you close its window. `brainslice-roi count` on
  such a file stops with "has no 'image_shape'" and says this.

## What it writes

- `<registry>/exports/<project>_ROI_Analysis/<animal>/<region>/<sample>_roi_counts.csv`
  -- the counts per ROI for each sample;
- `<registry>/exports/<project>_ROI_Analysis/roi_summary[_<REGION>].csv`
  -- one row per sample (the TOTAL row of each);
- one registry entry per sample (`<sample>__roi_counts` in that analysis's
  `registry.json`).

If no registry is configured (see [ANALYSIS_REGISTRY.md](ANALYSIS_REGISTRY.md)),
the CSVs go to `<dataset folder>/roi_counts/` instead (same layout: per-sample
CSVs under `<animal>/<region>/`, the summary at the top) and nothing is
registered; the command says so and prints the folder. Older versions wrote
them to `<dataset folder>/batch_results/`; that folder is searched for
detection results, so the counts no longer go there.

## How the counts stay tied to the measurements they were counted against

Each sample's registry entry records the `measurements.csv` it counted, that
file's SHA-256 fingerprint, and the method of the detection run that made it
(copied from the sample's entry in `<project>_Detection`). So when the
**approved method** changes and the detection is re-run under it, the old ROI
counts show as **stale** in
`mousebrain-registry --name <project>_ROI_Analysis --stale` until you count
again.

**Know this before trusting "current":** the counts are checked against the
detection run only at the moment they are counted. The detection batch
records, in each of its entries, the settings it actually ran with; so a
detection re-run with other settings (a different `--threshold`, say) makes
the NEW detection entry stale against the approved method, but the ROI
counts already registered keep the method of the run they were counted
against and are not looked at again. After re-running the detection for a
sample, count that sample again: the new count is tied to the new run.

The counts are recorded as **not tied** -- and always show as stale -- when
the measurements counted are not the ones the detection run registered: the
sample has no detection entry, the entry was invalidated, its registered
measurements file is missing, or that file's contents differ from the file
counted (for example, counted from an old `batch_results` copy). The
command lists each such sample with the reason. Run the detection batch for
those samples and count again.

If `approved_method.json` gives `<project>_Detection` its own entry, give
`<project>_ROI_Analysis` the same one; the command warns when the two
analyses are judged by different methods, because every ROI count would
then read as stale.
