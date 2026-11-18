# Receiving a new brain

For the person putting a newly imaged brain into the pipeline. No codebase knowledge
assumed. Nothing here is specific to one lab's storage or imaging partner; where your
own locations and settings belong is noted at the end.

## The short version

1. Put the `.ims` file somewhere other than the brains folder first.
2. Check whether it is a production scan or a low-resolution overview.
3. Name it `NUMBER_PROJECT_COHORT_ANIMAL_MAGx_zSTEP.ims`.
4. Drop it in the brains folder and run `1_organize_pipeline.py`.
5. Check the script lists it as ready. If it does not appear at all, stop.

Steps 2 and 5 are the ones people skip, and they are the two that cost real work.

## Why the file is staged somewhere else first

The final folder name contains the imaging parameters, so the name cannot be settled
until you know whether this is a production scan or an overview. Staging first also
means a half-finished transfer never appears in the brains folder looking like a brain.

So the order is: download, identify, then name and move.

**The imaging parameters are whatever the imaging was done at, as reported to you.**
Record them consistently and move on. They are a label, not a measurement to audit --
what matters downstream is that the same settings are named the same way every time, so
that brains imaged alike sort together.

## Telling a production scan from an overview

The one distinction worth checking in the file itself is categorical, not a matter of
decimals: an overview is sampled at roughly **double** the voxel size in every axis and
is several times smaller on disk. That is obvious at a glance and is not a judgement
about anyone's technique.

An `.ims` file is HDF5. The voxel size is the extent divided by the number of voxels,
from `DataSetInfo/Image`:

```python
import h5py

def voxel_size(path):
    """(x, y, z) voxel size in microns from an Imaris .ims file."""
    with h5py.File(path, "r") as h:
        img = h["DataSetInfo/Image"]
        def a(name):
            v = img.attrs[name]
            if hasattr(v, "__iter__") and not isinstance(v, (bytes, str)):
                v = b"".join(x if isinstance(x, bytes) else str(x).encode() for x in v)
            return float(v.decode() if isinstance(v, bytes) else v)
        n = {d: int(a(d) if d in img.attrs else a("Size" + d)) for d in "XYZ"}
        return ((a("ExtMax0") - a("ExtMin0")) / n["X"],
                (a("ExtMax1") - a("ExtMin1")) / n["Y"],
                (a("ExtMax2") - a("ExtMin2")) / n["Z"])
```

How to use it, step by step (MouseBrain has no command that reads the voxel size of
a file outside the brains folder, so this is the way):

1. Open an Anaconda Prompt and activate the MouseBrain environment
   (`conda activate <your MouseBrain environment>`); it already has `h5py`.
2. Type `python` and press Enter. The prompt changes to `>>>`.
3. Paste the block above and press Enter twice (until `>>>` comes back).
4. Type `voxel_size(r"<full path to the .ims file>")` and press Enter. It prints
   three numbers: the voxel size in micrometers along x, y and z.
5. Do the same for the other file you are comparing with, then type `exit()`.

If it stops with an error (`KeyError`, `OSError`), the file is not an Imaris file with the usual header
(or the transfer is incomplete); do not put it in the brains folder.

Without any of that, the file size already tells you most of it: in File Explorer, an
overview is several times smaller than a production scan of the same kind of brain.

(`util_ims_metadata_dump.py` in `scripts/utils` also prints the voxel size, but it only
lists `.ims` files that are already inside the brains folder, and it writes a
`<file name>_metadata.txt` beside the file, so it is no use on a file you have staged
elsewhere.)

Compare the two deliveries against each other, or against a brain you already have.
A factor of about two in every axis means an overview; the same ballpark means a
production scan. That is the whole question -- small variation between brains is normal
and not something this pipeline tries to detect or correct for.

## Overview scans are not pipeline inputs

Imaging partners often send a quick low-resolution overview alongside (or before) the
real scan. These are useful for checking that a brain is worth imaging properly, and
they must not be processed as production data: detection settings tuned at one voxel
size produce wrong counts at another, silently.

They are easy to recognise once the header is read -- the voxel size is roughly double
in every axis, and the file is several times smaller. A production brain and an
overview of the same specimen differ by a large factor in size, so a delivery that is
unexpectedly small deserves a header check before anything else.

`1_organize_pipeline.py` refuses any file whose name ends in `PANO`, reports it as
`[skip]`, and tells you **not** to rename it. Renaming an overview into the production
form is the one mistake that pulls it into the pipeline, so the refusal is deliberate
and is not something to work around.

Keep them beside the production brains, in the same folder shape, so the specimen's
history is in one place.

## Naming, exactly

```
NUMBER_PROJECT_COHORT_ANIMAL_MAGx_zSTEP.ims
101_PROJ_01_02_2.5x_z5.ims
```

* `NUMBER` is the brain number, a single lab-wide sequence, not per animal.
* `PROJECT_COHORT_ANIMAL` identifies the animal and must match a real subject in the
  database. Check it; a transposed digit files a brain under the wrong mouse and
  nothing downstream will notice.
* `MAGx_zSTEP` comes from the header, not from memory.
* Overview scans end in `PANO` in place of the imaging parameters.

**The folder and the file spell numbers differently.** Folders replace the decimal
point with `p`; the file keeps the dot:

```
101_PROJ_01_02/101_PROJ_01_02_2p5x_z5/0_Raw_IMS/101_PROJ_01_02_2.5x_z5.ims
```

This is easy to get backwards, and a mismatch does not error -- a later path lookup
just finds nothing.

## Where a brain lives

```
1_Brains/
  <mouse folder>/                 101_PROJ_01_02
    <pipeline folder>/            101_PROJ_01_02_2p5x_z5
      0_Raw_IMS/                  the .ims goes here
      1_Extracted_Full/
      2_Cropped_For_Registration/
      3_Registered_Atlas/
      4_Cell_Candidates/
      5_Classified_Cells/
      6_Region_Analysis/
```

Two levels, and both carry the brain number. The mouse folder holds every imaging run
of that specimen, so one brain imaged twice has two pipeline folders side by side.

Let `1_organize_pipeline.py` create this. Hand-made folders are how a brain ends up in
a shape the pipeline cannot see.

## Running the organizer

In an Anaconda Prompt, activate the environment and go to the steps' folder first:

```
conda activate <your MouseBrain environment>
cd <path to the MouseBrain repository>\scripts\pipeline
```

Then:

```
python 1_organize_pipeline.py --inspect    # say what would happen, change nothing
python 1_organize_pipeline.py              # do it
python 1_organize_pipeline.py --yes        # do it without asking (scripted or logged runs)
```

It prints one line per brain:

| line | meaning |
|---|---|
| `[OK] mouse/pipeline/ - ready for processing` | in place, ready for Script 2 |
| `[ ] ... - not yet organized` | found, will be moved when you run without `--inspect` |
| `[skip] ... PANO overview scan` | correct and deliberate; leave it alone |
| `[FAIL] ... doesn't match pattern` | the name is wrong; fix the name |

**Then run `--inspect` again and confirm your brain is listed as ready.** This is the
check that matters. A brain that appears in neither the ready list nor the needs-work
list is not organized -- it is in a shape the scanner cannot see, and it will sit there
untouched through every later step. (This was a real failure: a large brain was placed
flat, disappeared from the listing, and nothing reported a problem.)

**Old-layout folders.** Older layouts named a brain's folder by its imaging setup
only (`2p5x_z5` instead of `101_PROJ_01_02_2p5x_z5`) and used older names for the
numbered folders. The organizer renames these to the current names, merging into the
current folder when both exist. It never deletes anything it did not move: an item is
moved when the current folder has nothing by that name (or only an empty folder by
that name), and the old folder is removed only once it is empty. If something could
not be merged because the current folder already has its own copy, the old folder is
kept and the organizer prints `[!] Kept old short-name <folder>/: ...` with the names
left in it. Compare those by hand, move what you want to keep, and delete the old
folder yourself. (Before this, the merge deleted the old folder whatever was left in
it, which could take the brain's raw `.ims` with it.)

Folders whose names start with `_` are never searched, so a superseded delivery kept
in one (for example `_superseded_<date>`) is never moved into a brain. See
"Which folders are brains?" in `README.md`.

## After the organizer: what happens next, and what can wait

```
python 2_extract_and_analyze.py            # turn the .ims into TIFFs
```

By default this extracts only, and leaves cropping to you. Extraction is the
expensive, unattended part; cropping is a judgement call about where the brain
ends and the cord begins, and a bad crop is not obvious until registration
fails.

From here the order is: crop, then register (`3_register_to_atlas.py`), then
approve the registration QC by eye, then detect, classify, and count.

**Detection does not have to wait for all of that.** If registration is held up
-- a better scan is coming, the crop is not made, nobody has reviewed the QC --
you can run detection on the whole extracted stack now:

```
python 4_detect_cells.py --brain <brain_id> --source full
```

That gives you a cell count and candidates to look at, but no counts per brain
region, and the run has to be repeated on the crop later. The full explanation
of why, and where the results are kept so they cannot be mistaken for the real
ones, is in `README.md` under "Running detection before registration".

## Where your own details go

Everything above is true of any installation. The parts that are not -- who sends your
brains and how, which machine and drive they land on, which imaging settings your
detection is calibrated for, what your remote is called -- are properties of one lab.
Keep them in your own notes or configuration, not in this repository, and set
`CONNECTOME_ROOT` to the folder that CONTAINS `Tissue/MouseBrain_Pipeline` (not the
pipeline folder itself) so the tools know where everything is.
