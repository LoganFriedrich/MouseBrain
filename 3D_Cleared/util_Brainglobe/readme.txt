# util_Brainglobe - BrainGlobe Resources

**Location:** `\\134.48.78.185\blac\2_Connectome\3_Nuclei_Detection\util_Brainglobe\`  
**Last Updated:** 2025-12-16

---

## Purpose

Resources for the BrainGlobe pipeline: trained neural networks, preprocessing macros, and development/calibration data.

**Scripts live in:** `../util_Scripts/` (sibling folder)  
**Brain data lives in:** `../1_Brains/` (sibling folder)

---

## Folder Structure

```
util_Brainglobe/
│
├── Trained_Models/              ← Cellfinder neural networks
│   ├── 20241007/
│   ├── 20251212/
│   └── ...
│
├── 1_ImageJ_Macros/             ← Preprocessing macros (sharpening, background)
│
└── _Pipeline_Development/       ← Test/calibration data (not for analysis)
    ├── 143/
    ├── 167/
    ├── 168/
    └── From_Imaris/
```

---

## Brain Data Location

Brain folders live in `../1_Brains/`:

```
3_Nuclei_Detection/
├── util_Scripts/            ← Pipeline scripts
├── util_Brainglobe/         ← You are here (resources)
├── 1_Brains/                ← Brain data lives here
│   ├── 349_CNT_01_02/
│   └── 356_CNT_01_13/
└── 2_Summary_Data/
```

---

## Brain Folder Naming

**Format:** `{ID}_{PROJECT}_{COHORT}_{ANIMAL}`

| Component | Meaning | Example |
|-----------|---------|---------|
| ID | Unique brain number | `349` |
| PROJECT | Project code | `CNT` (control), `SCI` (injury) |
| COHORT | Cohort number | `01` |
| ANIMAL | Animal within cohort | `02` |

**Example:** `349_CNT_01_02` = Brain #349, Control project, Cohort 1, Animal 2

---

## Inside Each Brain Folder

Each brain has this structure:

```
349_CNT_01_02/
│
├── 0_Original/                      ← Raw IMS files (moved here by Script 1)
│   └── 349_CNT_01_02_1p625x_z4.ims
│
└── 1p625x_z4/                       ← Pipeline per magnification
    ├── 1_Channels/                  ← Extracted TIFF slices
    │   ├── ch0/                     ← Channel 0 (autofluorescence)
    │   │   ├── Z0000.tif
    │   │   └── ...
    │   ├── ch1/                     ← Channel 1 (signal)
    │   └── metadata.json
    ├── 2_Registration/              ← brainreg output
    ├── 3_Detection/                 ← cellfinder output
    └── 4_Analysis/                  ← Final results for this brain
```

### Magnification Folders

Named from the IMS filename with decimals converted to `p`:
- `1.625x_z4` → `1p625x_z4`
- `1.9x_z3.37` → `1p9x_z3p37`

This avoids issues with periods in folder names (napari compatibility).

### Numbered Subfolders

| Folder | Created By | Contains |
|--------|------------|----------|
| `1_Channels/` | Script 1 | Extracted TIFF slices per channel |
| `2_Registration/` | Script 2 (brainreg) | Atlas alignment files |
| `3_Detection/` | Script 3 (cellfinder) | Detected cell coordinates |
| `4_Analysis/` | Script 4+ | Brain-specific results, figures |

---

## IMS File Naming

**Required format:** `{ID}_{PROJECT}_{COHORT}_{ANIMAL}_{MAG}x_z{STEP}.ims`

**Example:** `349_CNT_01_02_1.625x_z4.ims`

| Component | Meaning |
|-----------|---------|
| `1.625x` | Magnification (used to calculate XY voxel size) |
| `z4` | Z-step in microns |

**Important:** Rename files to this format BEFORE running the pipeline. The scripts extract voxel information from the filename.

---

## Pipeline Workflow

```
1. Create brain folder in ../1_Brains/ (e.g., 349_CNT_01_02/)
2. Place IMS files in brain folder (loose, not in subfolder)
3. Run Script 1 from ../util_Scripts/ → Creates 0_Original/, moves IMS, extracts channels
4. Run Script 2 → Registers to atlas, outputs to 2_Registration/
5. Run Script 3 → Detects cells, outputs to 3_Detection/
6. Run Script 4 → Analyzes results, outputs to 4_Analysis/
```

---

## Rules

1. **One brain per folder** - don't mix multiple animals
2. **Rename IMS files before processing** - scripts depend on filename format
3. **Don't modify `1_Channels/` after creation** - downstream steps depend on it
4. **Multiple magnifications are fine** - each gets its own pipeline subfolder
5. **`_Pipeline_Development/` is for testing only** - don't use that data for analysis

---

## Scripts Reference

Scripts live in `../util_Scripts/`. Run from that folder:

```bash
# Convert IMS to TIFF (run first)
python 1_ims_to_brainglobe.py

# Preview without processing
python 1_ims_to_brainglobe.py --inspect

# Process specific brain folder
python 1_ims_to_brainglobe.py "..\1_Brains\349_CNT_01_02"

# Nuke thumbs.db files (if napari drag-drop fails)
python util_thumbs_destroyer.py "..\1_Brains"
```

**Default data path:** Scripts look in `..\1_Brains\`