# Data Hierarchy

## Structure (smallest to largest)

```
Brain (374)
  └── Subject (CNT_01_01)
        └── Cohort (CNT_01, subjects 01-16)
              └── Experiment (CNT_01-04_01-16 = cohorts 1-4, 16 subjects each)
                    └── Project (CNT = Connectome, ENCR = Enhancer, etc.)
```

## Definitions

| Level | Example | Description |
|-------|---------|-------------|
| **Brain** | 374 | Individual tissue sample, has a numeric ID |
| **Subject** | CNT_01_01 | The animal/unit of study. Format: `{PROJECT}_{COHORT}_{SUBJECT#}` |
| **Cohort** | CNT_01 | Group processed together with same treatments. Contains multiple subjects (e.g., 01-16) |
| **Experiment** | CNT_01-04_01-16 | Scientific experiment, may span multiple cohorts |
| **Project** | CNT, ENCR | Umbrella research project |

## Naming Convention

`{BRAIN#}_{PROJECT}_{COHORT}_{SUBJECT#}`

Example: `374_CNT_01_05` = Brain 374, from Connectome project, Cohort 01, Subject 05

## Project Codes

- **CNT** = Connectome
- **ENCR** = Enhancer
- (add others as needed)

## Analysis Modes

1. **Calibration** - Tuning detection parameters on representative samples to find optimal settings
2. **Utilization** - Applying validated settings to generate experimental data for publication

Calibration documents HOW settings were optimized.
Utilization generates the actual scientific results.
