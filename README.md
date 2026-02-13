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

## Acknowledgments

This tool builds on excellent open-source neuroscience software:

- **[BrainGlobe](https://brainglobe.info/)** - Atlas registration, segmentation, and the atlas API ([GitHub](https://github.com/brainglobe))
  > Claudi, F., Petrucco, L., et al. (2020). BrainGlobe Atlas API: a common interface for neuroanatomical atlases. *JOSS*, 5(54), 2668.
- **[cellfinder](https://brainglobe.info/documentation/cellfinder/)** - Whole-brain cell detection and classification ([GitHub](https://github.com/brainglobe/cellfinder))
  > Tyson, A.L., Velez-Fort, M., et al. (2021). Accurate determination of marker-positive cell bodies in tissue sections. *Scientific Reports*, 11, 21505.
