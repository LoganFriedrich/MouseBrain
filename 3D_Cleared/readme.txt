# 3_Nuclei_Detection

**Location:** `\\134.48.78.185\blac\2_Connectome\3_Nuclei_Detection\`  
**Last Updated:** 2025-12-16

---

## Purpose

Nuclei detection from cleared brain tissue using lightsheet microscopy.

---

## Folder Structure

```
3_Nuclei_Detection/
│
├── util_Scripts/            ← Pipeline scripts (Python)
│   ├── 1_ims_to_brainglobe.py
│   ├── util_thumbs_destroyer.py
│   └── ...
│
├── util_Brainglobe/         ← BrainGlobe tool resources
│   ├── Trained_Models/      ← Cellfinder neural networks
│   ├── 1_ImageJ_Macros/     ← Preprocessing macros
│   └── _Pipeline_Development/  ← Test/calibration data (not for analysis)
│
├── 1_Brains/                ← Individual brain data
│   ├── 349_CNT_01_02/
│   └── 356_CNT_01_13/
│
└── 2_Data_Summary/          ← Aggregated results across brains
```

---

## Naming Convention

| Prefix | Meaning | Examples |
|--------|---------|----------|
| `#_Name/` | **Data folder** - numbered by workflow | `1_Brains/`, `2_Data_Summary/` |
| `util_Name/` | **Utility folder** - tools/resources | `util_Scripts/`, `util_Brainglobe/` |
| `_Name/` | **System/meta** - archives, development | `_Pipeline_Development/` |

---

## Contents

| Folder | Purpose |
|--------|---------|
| `util_Scripts/` | Python scripts for the processing pipeline |
| `util_Brainglobe/` | BrainGlobe resources: trained models, ImageJ macros, development data |
| `1_Brains/` | One folder per brain containing full processing pipeline |
| `2_Data_Summary/` | Cross-brain comparisons, group statistics, summary figures |

---

## Workflow

1. Place IMS files in a brain folder under `1_Brains/`
2. Run scripts from `util_Scripts/` to process
3. Each brain folder builds out its own pipeline structure
4. Aggregate results go to `2_Data_Summary/`

See `util_Brainglobe/_README.txt` for detailed brain folder organization and pipeline steps.