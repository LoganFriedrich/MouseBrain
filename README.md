# MouseBrain

Unified package for Connectome tissue analysis tools.

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

Then in napari: **Plugins → Connectome Pipeline → 2. Setup & Tuning**

## What's Included

- **Connectome Pipeline** - napari plugin for cell detection workflow
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
  instead of `approved_method.json` under the registry root.
- MouseBrain never pushes results anywhere. An integrator (a lab database such
  as mousedb, for example) may pull from this folder; the manifest tells it
  what is there, how it was produced and whether it is still current.
- Inspect a registry: `mousebrain-registry --name <analysis> --stale --summary`.

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
