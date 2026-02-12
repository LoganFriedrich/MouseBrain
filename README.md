# MouseBrain

Unified package for Connectome tissue analysis tools.

## Installation

```powershell
# 1. Open PowerShell and activate the environment
conda activate Y:\2_Connectome\envs\MouseBrain

# 2. Go to this folder and run the installer
cd Y:\2_Connectome\Tissue\MouseBrain
.\install.ps1
```

That's it. The script installs everything correctly.

## Usage

```bash
mousebrain              # Launch napari with all plugins
mousebrain --check      # Verify everything is working
mousebrain --paths      # Show configured paths
```

Then in napari: **Plugins → Connectome Pipeline → 2. Setup & Tuning**

## What's Included

- **Connectome Pipeline** - napari plugin for cell detection workflow
- **BrainGlobe** - Atlas registration (brainreg)
- **Cellfinder** - Cell detection and classification
- **Experiment Tracker** - Track all calibration runs

## Troubleshooting

If something breaks, re-run the installer:
```powershell
.\install.ps1
```

Or check what's wrong:
```powershell
mousebrain --check
```
