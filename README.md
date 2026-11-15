# MouseBrain

Unified package for whole-brain (3D cleared) and slice (2D) tissue analysis tools.

## Installation

```powershell
# 1. Open PowerShell and activate the environment
conda activate MouseBrain    # or the full path of the environment if it was created with --prefix

# 2. Go to this folder and run the installer
cd <path to this repository>
.\install.ps1
```

That's it. The script installs everything correctly.

## Usage

```bash
mousebrain              # Launch napari with all plugins
mousebrain --check      # Verify everything is working
mousebrain --paths      # Show configured paths
```

Then in napari: **Plugins → BrainTools → 3D: 2. Setup & Tuning**

## What's Included

- **BrainTools** - napari plugin (Plugins → BrainTools) for the 3D cell detection workflow and 2D slice analysis
- **BrainGlobe** - Atlas registration (brainreg)
- **Cellfinder** - Cell detection and classification
- **Experiment Tracker** - Track all calibration runs

## Where outputs go

Every analysis output (detection measurements, ROI counts, figures) is recorded
by the analysis registry together with the parameters, source files and time
that produced it. The registry lives inside the pipeline data folder:

```
<pipeline root>/Registry/
├── exports/<analysis>/registry.json   # manifest: one entry per output, with provenance
├── exports/<analysis>/...             # data files (CSV)
├── figures/<analysis>/...             # figures, by animal/region
├── logs/<analysis>.log                # audit trail
└── approved_method.json               # optional: the parameters your lab has approved
```

- The pipeline root comes from `CONNECTOME_ROOT` (`<root>/Tissue/MouseBrain_Pipeline`),
  or is auto-detected when the package is installed inside such a layout. Set
  `MOUSEBRAIN_REGISTRY_ROOT` to put the registry somewhere else. If neither
  resolves, nothing is written and the tool stops with a message naming them.
- `MOUSEBRAIN_APPROVED_METHOD` may point to a JSON file of method parameters
  instead of `approved_method.json` under the registry root. The file may give
  each analysis its own approved method; analyses that record their own
  parameters (3D region counts, orientation checks) have none unless it names
  them, and are never reported stale. After changing it, run
  `mousebrain-registry --name <analysis> --refresh-approved` for each analysis.
- Two machines may register at once (a lock file serialises the writes); a
  damaged manifest is set aside as `registry.json.unreadable-<time>`, never
  emptied. Full guide: [docs/pipeline/ANALYSIS_REGISTRY.md](docs/pipeline/ANALYSIS_REGISTRY.md).
- MouseBrain never pushes results anywhere. An integrator (a lab database such
  as mousedb, for example) may pull from this folder; the manifest tells it
  what is there, how it was produced and whether it is still current.
- Inspect a registry: `mousebrain-registry --name <analysis> --stale --summary`.
- Invalidating an entry moves its files to `_archived/<timestamp>/` and marks
  it not current (with `invalidated_reason` when the caller gave one, and
  `archived_outputs` saying where each file went). When the same sample is
  registered again (`register_output` or `register_roi_counts`), the
  invalidated record is kept inside the new entry under `superseded` (newest
  first), with its `outputs` pointing at the archived files, so the manifest
  still shows what the sample's earlier result was, why it was retired and
  where its files are.

## Annotating ND2 slices (2D: BrainSlice, tab "7. Annotate")

Open **Plugins → BrainTools → 2D: 1. BrainSlice** and go to the **7. Annotate**
tab. It has its own tabs: Load, Channels, Annotate, Export.

**Loading.** Click **Load ND2 File...** (one file) or **Load Folder...** (every
`.nd2` in a folder, one tissue section per file). While a file loads, the load
button is grayed out and a progress bar runs.

- *Changing your mind mid-load:* starting another load (for example by
  double-clicking a file under Recent Files) cancels the one that is running.
  The cancelled load is dropped silently: no error box appears, and the load
  button stays grayed out until the NEW file has finished loading. The window
  may pause briefly while the old load stops.
- *A file that really fails to load* still shows a "Load Error" box with the
  reason, and the buttons come back on so you can try another file.

**Turning the image (Load tab, Transform: Rotate / Flip).** The rotate and
flip buttons turn every channel of every section. Anything you drew with the
**Annotate** tab's buttons (shapes, arrows, text boxes, text labels) turns
with the image, so each annotation stays on the feature it marks. Your
channel settings are kept: contrast, color, gamma, opacity and which
channels are hidden do not change when you rotate or flip, whether you set
them in the Channels tab or in napari's own layer controls.

- If you closed one of the channel layers, rotate/flip brings it back. The
  channels you did not close keep their settings; the one you closed comes
  back with the default settings.
- Only layers made with the Annotate tab's buttons move. A shapes or points
  layer you added through napari's own "new layer" buttons is not touched;
  draw annotations with the Annotate tab if they must follow the image.

**Where exports go (Export tab, Output).** Each time you load a file, the
output folder is set to that file's own folder (for **Load Folder...**, the
folder you loaded). Loading another file moves it to that file's folder, so
pictures never land beside a different file's images.

- To export somewhere else, type a folder into the box or click
  **Browse...**. Your folder is then kept for every later load.
- To go back to "the loaded file's folder", empty the box; the next load
  fills it in again.

## Troubleshooting

If something breaks, re-run the installer:
```powershell
.\install.ps1
```

Or check what's wrong:
```powershell
mousebrain --check
```

## Acknowledgments

This tool builds on excellent open-source neuroscience software:

- **[BrainGlobe](https://brainglobe.info/)** - Atlas registration, segmentation, and the atlas API ([GitHub](https://github.com/brainglobe))
  > Claudi, F., Petrucco, L., et al. (2020). BrainGlobe Atlas API: a common interface for neuroanatomical atlases. *JOSS*, 5(54), 2668.
- **[cellfinder](https://brainglobe.info/documentation/cellfinder/)** - Whole-brain cell detection and classification ([GitHub](https://github.com/brainglobe/cellfinder))
  > Tyson, A.L., Velez-Fort, M., et al. (2021). Accurate determination of marker-positive cell bodies in tissue sections. *Scientific Reports*, 11, 21505.
