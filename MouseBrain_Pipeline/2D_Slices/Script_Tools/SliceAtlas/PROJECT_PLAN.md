# SliceAtlas: 2D Brain Slice Registration & Cell Counting Pipeline

## Project Goal
Automated pipeline to:
1. Register 2D brain slice images to a standard atlas (e.g., Allen CCF)
2. Detect and segment brain regions
3. Count cells within each region

**Secondary Goal**: Learn Python programming through building this real-world application.

## Learning Path

This project will teach Python concepts progressively:

### Level 1: Python Fundamentals
- [ ] Variables, types, and basic operations
- [ ] Functions and control flow (if/for/while)
- [ ] Lists, dictionaries, and data structures
- [ ] File I/O basics
- [ ] Imports and modules

### Level 2: Scientific Python
- [ ] NumPy arrays (the foundation of image data)
- [ ] Array indexing and slicing
- [ ] Basic image operations with scikit-image
- [ ] Pandas for data tables
- [ ] Matplotlib for visualization

### Level 3: Object-Oriented Python
- [ ] Classes and objects
- [ ] Methods and attributes
- [ ] Inheritance basics
- [ ] When to use classes vs functions

### Level 4: Application Development
- [ ] Project structure and packaging
- [ ] napari widget development
- [ ] Working with external libraries (ABBA, etc.)
- [ ] Error handling and debugging
- [ ] Testing basics

## Recommended Tool Stack

### Primary Tools (napari-compatible)

| Tool | Purpose | Why |
|------|---------|-----|
| **ABBA** (napari-abba) | 2D slice registration | Gold standard for slice-to-atlas registration, excellent napari plugin |
| **DeepSlice** | Initial pose estimation | AI-based, speeds up ABBA workflow significantly |
| **brainglobe-atlasapi** | Atlas access | Unified API for Allen CCF, Waxholm, etc. |
| **cellfinder** / **stardist** | Cell detection | Proven pipelines, napari plugins available |
| **napari** | Visualization & QC | Central interface for all steps |

### Alternative Approaches
- **QuPath + ABBA**: If you prefer QuPath's cell detection
- **brainreg**: If you later need 3D whole-brain registration
- **Custom napari widgets**: For pipeline-specific QC steps

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT IMAGES                              │
│              (TIFF, OME-TIFF, or similar)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: PREPROCESSING                                          │
│  - Load images                                                   │
│  - Channel selection (DAPI, markers, etc.)                      │
│  - Basic corrections (illumination, background)                  │
│  napari widget: preprocessing_widget.py                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: ATLAS REGISTRATION                                     │
│  - DeepSlice for initial alignment (optional, speeds things up) │
│  - ABBA for precise registration                                │
│  - Manual refinement in napari-abba                             │
│  napari widget: registration_widget.py (wraps napari-abba)      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: REGION EXTRACTION                                      │
│  - Apply registration transforms                                 │
│  - Extract atlas labels per pixel                               │
│  - Generate region masks                                        │
│  napari widget: regions_widget.py                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: CELL DETECTION                                         │
│  - Detect cells (StarDist, Cellpose, or cellfinder)            │
│  - Filter by size, intensity, etc.                              │
│  - Optional: classify cell types                                │
│  napari widget: detection_widget.py                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: QUANTIFICATION                                         │
│  - Assign cells to regions                                      │
│  - Count cells per region                                       │
│  - Export statistics (CSV, JSON)                                │
│  napari widget: quantification_widget.py                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUTS                                   │
│  - Registered images                                            │
│  - Region annotations                                           │
│  - Cell coordinates + region assignments                        │
│  - Summary statistics per region                                │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure (Planned)

```
SliceAtlas/
├── PROJECT_PLAN.md          # This file
├── TOOLS.md                 # Detailed tool documentation
├── DECISIONS.md             # Architecture decisions log
├── environment.yml          # Conda environment
├── pyproject.toml           # Package configuration
│
├── sliceatlas/              # Main package
│   ├── __init__.py
│   ├── core/                # Core functionality
│   │   ├── io.py            # Image loading/saving
│   │   ├── registration.py  # ABBA/DeepSlice wrappers
│   │   ├── detection.py     # Cell detection
│   │   └── quantification.py
│   │
│   ├── napari_widgets/      # napari plugin widgets
│   │   ├── __init__.py
│   │   ├── napari.yaml      # Plugin manifest
│   │   ├── preprocessing_widget.py
│   │   ├── registration_widget.py
│   │   ├── regions_widget.py
│   │   ├── detection_widget.py
│   │   └── quantification_widget.py
│   │
│   └── cli/                 # Command-line interface (optional)
│       └── main.py
│
├── tests/
└── examples/
    └── sample_workflow.py
```

## Key Technical Decisions to Make

1. **Atlas choice**: Allen CCF (mouse)? Waxholm (rat)? Others?
2. **Cell detection method**: StarDist vs Cellpose vs cellfinder
3. **Image format**: What format are your source images?
4. **Batch processing**: Single slices vs entire slide series?
5. **Classification**: Just counting, or cell type classification?

## Development Phases

### Phase 1: Core Infrastructure
- [ ] Set up environment with ABBA, napari, brainglobe
- [ ] Test napari-abba installation and basic workflow
- [ ] Create basic image loading utilities

### Phase 2: Registration Pipeline
- [ ] Integrate DeepSlice for initial pose
- [ ] Wrapper for ABBA registration
- [ ] Export registered coordinates/transforms

### Phase 3: Cell Detection
- [ ] Integrate StarDist or Cellpose
- [ ] Cell filtering and QC widget
- [ ] Region assignment logic

### Phase 4: Quantification & Export
- [ ] Cell counting per region
- [ ] Statistics and visualization
- [ ] Batch processing support

### Phase 5: Polish
- [ ] End-to-end napari plugin
- [ ] Documentation
- [ ] Example datasets

## References

- [ABBA Documentation](https://biop.github.io/ijp-imagetoatlas/)
- [napari-abba](https://github.com/BIOP/napari-abba)
- [DeepSlice](https://github.com/PolarBean/DeepSlice)
- [BrainGlobe](https://brainglobe.info/)
- [StarDist](https://github.com/stardist/stardist)
