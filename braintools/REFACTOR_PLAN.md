# braintools Package Refactor Plan

> **Status**: Planning Document (No Code Changes Yet)
> **Last Updated**: 2025-01-26
> **Scope**: Future consolidation of SCI-Connectome pipeline scripts into unified package

---

## Executive Summary

The SCI-Connectome pipeline currently consists of 25+ standalone scripts scattered across `Tissue/3D_Cleared/util_Scripts/`. This refactor plan outlines a phased approach to consolidate these scripts into the `braintools` package, improving maintainability, discoverability, and reproducibility while preserving backward compatibility.

**Key Goals:**
- Centralize pipeline logic into a single installable package
- Establish clear module hierarchy (pipeline → utils → analysis)
- Enable CLI-based workflow without directory changes
- Maintain napari plugin as separate installable
- Preserve experiment tracking and reproducibility

---

## Current State

### Package Structure
```
braintools/
├── src/braintools/
│   ├── __init__.py
│   └── cli.py                    # Basic napari launcher
├── pyproject.toml
└── README.md
```

### Pipeline Scripts (`Tissue/3D_Cleared/util_Scripts/`)

#### Core Pipeline (6 scripts)
| Script | Purpose | Status |
|--------|---------|--------|
| `1_organize_pipeline.py` | Create folder structure, parse brain names | Active |
| `2_extract_and_analyze.py` | Extract channels, auto-crop | Active |
| `3_register_to_atlas.py` | BrainGlobe registration | Active |
| `4_detect_cells.py` | Cell candidate detection | Active |
| `5_classify_cells.py` | Classification post-processing | Active |
| `6_count_regions.py` | Count cells per anatomical region | Active |

#### Infrastructure (3 scripts)
| Script | Purpose |
|--------|---------|
| `config.py` | Path configuration, brain name parsing |
| `experiment_tracker.py` | Calibration run logging to CSV |
| `RUN_PIPELINE.py` | Orchestrator for sequential execution |

#### Utilities (15+ scripts)
| Category | Scripts |
|----------|---------|
| **Registration** | `util_approve_registration.py`, `util_registration_qc.py` |
| **Cropping** | `util_manual_crop.py`, `util_optimize_crop.py` |
| **Training** | `util_create_training_cubes.py` |
| **Imaging** | `util_ims_metadata_dump.py`, `util_estimate_voxel_size.py` |
| **Batch Processing** | `batch_process.py`, `overnight_batch.py` |
| **Analysis** | `util_compare_to_published.py`, `elife_region_mapping.py`, `batch_train_compare.py` |
| **External Data** | `util_fetch_elife_cervical.py`, `util_generate_elife_report.py` |

#### Brain-Specific Scripts (cleanup candidates)
- `run_368_count.py`
- `run_368_simple.py`
- `direct_368_count.py`

#### Duplicates / Archived
- `util_DevScripts/` - Contains duplicate utilities

### napari Plugin

```
sci_connectome_napari/
├── setup.py / pyproject.toml
├── sci_connectome_napari/
│   ├── __init__.py
│   ├── tuning_widget.py         # Main widget (~5600 lines)
│   ├── pipeline_widget.py
│   ├── manual_crop_widget.py
│   ├── experiments_widget.py
│   ├── curation_widget.py
│   └── session_documenter.py
```

---

## Target State

### New Package Structure

```
braintools/
├── src/braintools/
│   ├── __init__.py
│   ├── cli.py                           # Unified CLI orchestrator
│   │
│   ├── config.py                        # Path config (moved)
│   ├── tracker.py                       # Experiment tracker (moved)
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── organize.py                  # 1_organize_pipeline
│   │   ├── extract.py                   # 2_extract_and_analyze
│   │   ├── register.py                  # 3_register_to_atlas
│   │   ├── detect.py                    # 4_detect_cells
│   │   ├── classify.py                  # 5_classify_cells
│   │   ├── count.py                     # 6_count_regions
│   │   ├── batch.py                     # Batch processing (moved)
│   │   └── orchestrator.py              # RUN_PIPELINE merge
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── registration.py              # approve + QC utils
│   │   ├── crop.py                      # manual + optimize crop
│   │   ├── training.py                  # training cube generation
│   │   ├── imaging.py                   # IMS metadata, voxel size
│   │   └── io.py                        # Common I/O utilities
│   │
│   └── analysis/
│       ├── __init__.py
│       ├── elife.py                     # eLIFE report generation
│       ├── comparison.py                # batch_train_compare
│       └── published_comparison.py      # compare_to_published
│
├── pyproject.toml                       # Updated with new entry points
├── REFACTOR_PLAN.md                     # This file
└── MIGRATION_STATUS.md                  # Track progress per tier
```

---

## Migration Strategy

### Principles

1. **Phased approach**: Complete one tier before starting next
2. **Preserve backward compatibility**: Old scripts remain until no references exist
3. **Test each migration**: Run pipeline after each tier completes
4. **Track progress**: Update `MIGRATION_STATUS.md` during work
5. **Keep napari plugin separate**: Remains installable independently

### Tier 1: Core Pipeline (First Priority)

**Rationale**: Core pipeline is the foundation; everything depends on it working correctly.

**Modules to migrate:**

| Current | New Location | Notes |
|---------|--------------|-------|
| `config.py` | `braintools.config` | Already works at package level; add path auto-detection for installed package |
| `experiment_tracker.py` | `braintools.tracker` | CSV path must be stable; may need `--tracker-dir` option |
| `1_organize_pipeline.py` | `braintools.pipeline.organize` | Entry function → `def main(args)` |
| `2_extract_and_analyze.py` | `braintools.pipeline.extract` | Entry function → `def main(args)` |
| `3_register_to_atlas.py` | `braintools.pipeline.register` | Entry function → `def main(args)` |
| `4_detect_cells.py` | `braintools.pipeline.detect` | Entry function → `def main(args)` |
| `5_classify_cells.py` | `braintools.pipeline.classify` | Entry function → `def main(args)` |
| `6_count_regions.py` | `braintools.pipeline.count` | Entry function → `def main(args)` |
| `RUN_PIPELINE.py` | Merge into `braintools.cli` | Add `--tier` option to run specific steps |

**CLI Entry Points (new):**
```toml
[project.scripts]
braintool-organize = "braintools.pipeline.organize:main"
braintool-extract = "braintools.pipeline.extract:main"
braintool-register = "braintools.pipeline.register:main"
braintool-detect = "braintools.pipeline.detect:main"
braintool-classify = "braintools.pipeline.classify:main"
braintool-count = "braintools.pipeline.count:main"
braintool-batch = "braintools.pipeline.batch:main"
braintool-run-all = "braintools.cli:run_pipeline"  # Replaces RUN_PIPELINE.py
```

**Testing Checklist:**
- [ ] Import all modules without errors
- [ ] All path calculations work from installed package location
- [ ] Tracker CSV created at expected location
- [ ] `braintool-organize` creates folder structure correctly
- [ ] Full pipeline completes on test brain
- [ ] Calibration runs logged to tracker

### Tier 2: Utilities (Second Priority)

**Rationale**: Utilities are used by users but not core pipeline; low risk to migrate after Tier 1 stable.

**Modules to migrate:**

| Current | New Location | Notes |
|---------|--------------|-------|
| `util_approve_registration.py` | `braintools.utils.registration` | Add `main()` function |
| `util_registration_qc.py` | `braintools.utils.registration` | Merge with above or keep separate |
| `util_manual_crop.py` | `braintools.utils.crop` | Both cropping utils go here |
| `util_optimize_crop.py` | `braintools.utils.crop` | Merge with above |
| `util_create_training_cubes.py` | `braintools.utils.training` | Add `main()` function |
| `util_ims_metadata_dump.py` | `braintools.utils.imaging` | Imaging utilities group |
| `util_estimate_voxel_size.py` | `braintools.utils.imaging` | Merge with above |
| `batch_process.py` | `braintools.pipeline.batch` | Batch orchestrator |
| `overnight_batch.py` | `braintools.pipeline.batch` | Merge with batch_process |

**CLI Entry Points (new):**
```toml
braintool-approve-registration = "braintools.utils.registration:approve_main"
braintool-qc-registration = "braintools.utils.registration:qc_main"
braintool-manual-crop = "braintools.utils.crop:manual_main"
braintool-optimize-crop = "braintools.utils.crop:optimize_main"
braintool-training-cubes = "braintools.utils.training:main"
braintool-ims-metadata = "braintools.utils.imaging:metadata_main"
braintool-estimate-voxel = "braintools.utils.imaging:voxel_main"
braintool-batch-process = "braintools.pipeline.batch:batch_main"
braintool-overnight-batch = "braintools.pipeline.batch:overnight_main"
```

**Testing Checklist:**
- [ ] All utilities import correctly
- [ ] CLI commands work with `--help`
- [ ] Core pipeline still works after migration
- [ ] No import conflicts

### Tier 3: Analysis & Cleanup (Third Priority)

**Rationale**: Analysis scripts are less frequently used; safe to migrate last.

**Modules to migrate:**

| Current | New Location | Status |
|---------|--------------|--------|
| `util_compare_to_published.py` | `braintools.analysis.published_comparison` | Active |
| `util_fetch_elife_cervical.py` | `braintools.analysis.elife` | Active |
| `util_generate_elife_report.py` | `braintools.analysis.elife` | Active |
| `elife_region_mapping.py` | `braintools.analysis.elife` | Active |
| `batch_train_compare.py` | `braintools.analysis.comparison` | Active |

**Scripts to Archive or Delete:**

| Script | Action | Reason |
|--------|--------|--------|
| `run_368_count.py` | Move to `Archive/` | Brain-specific; not general-purpose |
| `run_368_simple.py` | Move to `Archive/` | Brain-specific; not general-purpose |
| `direct_368_count.py` | Move to `Archive/` | Brain-specific; not general-purpose |
| `util_DevScripts/*` | Delete after audit | Check for duplicate logic |

**Testing Checklist:**
- [ ] All analysis modules import correctly
- [ ] Core pipeline still works
- [ ] Analysis scripts work independently
- [ ] Brain-specific scripts archived

---

## Implementation Details

### config.py Migration

**Current behavior:**
```python
# Auto-detects folder structure based on script location
BASE_PATH = pathlib.Path(__file__).parent.parent.parent
```

**After migration:**
```python
# For installed package, use environment variables + defaults
def get_base_path():
    if env.get('BRAINTOOLS_BASE_PATH'):
        return Path(env['BRAINTOOLS_BASE_PATH'])

    # Default to user home or current working directory
    return Path.home() / '2_Connectome'  # or configurable
```

**Action items:**
- Add `--base-path` CLI option to override
- Create `.braintools.config` file for persistent settings
- Document environment variable usage

### experiment_tracker.py Migration

**CSV location challenge:**
- Currently: `Tissue/3D_Cleared/2_Data_Summary/calibration_runs.csv`
- After migration: Must be stable and discoverable

**Solution:**
```python
def get_tracker_dir():
    if env.get('BRAINTOOLS_TRACKER_DIR'):
        return Path(env['BRAINTOOLS_TRACKER_DIR'])

    # Use base_path
    base = get_base_path()
    return base / 'Tissue' / '3D_Cleared' / '2_Data_Summary'
```

### napari Plugin Updates

**Required changes after Tier 1:**
```python
# Old import
from experiment_tracker import ExperimentTracker

# New import
from braintools.tracker import ExperimentTracker
```

**Files to update:**
- `tuning_widget.py` - Main widget
- `pipeline_widget.py` - Pipeline runner
- Other widgets using tracker or config

**Action items:**
- Update imports in all widget files
- Test napari plugin with moved modules
- Ensure plugin still auto-discovers runs

### CLI Consolidation (RUN_PIPELINE.py → cli.py)

**Current RUN_PIPELINE.py:**
```python
# Sequential orchestrator for all 6 steps
def run_full_pipeline(brain_path, start_step=1):
    ...
```

**After migration:**
```python
# In braintools/cli.py
@click.command()
@click.option('--brain', required=True)
@click.option('--start-step', default=1, type=int)
@click.option('--end-step', default=6, type=int)
@click.option('--base-path', envvar='BRAINTOOLS_BASE_PATH')
def run_pipeline(brain, start_step, end_step, base_path):
    """Run pipeline steps sequentially."""
    from braintools.pipeline import orchestrator
    orchestrator.run(brain, start_step, end_step, base_path)
```

---

## Prerequisites

### Before Starting Tier 1

- [ ] All tests pass on current scripts
- [ ] No active development on pipeline scripts
- [ ] Create `MIGRATION_STATUS.md` to track tier progress
- [ ] Set up staging branch for refactoring work

### Before Each Tier

- [ ] Review all imports in affected modules
- [ ] Identify any cross-dependencies
- [ ] Plan for deprecation warnings (if keeping old scripts)
- [ ] Document any behavior changes

### General Requirements

1. **Path stability**: Config and tracker must work from installed package
2. **No breaking changes**: Existing scripts should continue to work during transition
3. **Full testing**: Run complete pipeline after each tier
4. **Documentation updates**: Update claude.md files as migrations complete

---

## Timeline & Effort

| Tier | Modules | Estimated Effort | Duration |
|------|---------|------------------|----------|
| **Tier 1** | 8 core + 3 infrastructure | 3-4 days | Week 1 |
| **Tier 2** | 9 utilities | 2-3 days | Week 2 |
| **Tier 3** | 5 analysis + cleanup | 1-2 days | Week 3 |
| **Cleanup** | Remove old scripts, final testing | 1 day | Week 3 |

**Total: ~1 week of focused work**

---

## Rollback Plan

If migration fails at any tier:

1. **During development**: Keep both old and new code; old scripts untouched
2. **If Tier 1 fails**: Delete new package modules; revert imports; scripts continue as-is
3. **If Tier 2/3 fail**: Mark utility as incomplete; continue with core-only migration

**Key point**: No breaking changes; old scripts always remain functional.

---

## Success Criteria

### Tier 1 Complete
- [ ] All 6 pipeline steps runnable via `braintool-*` commands
- [ ] `braintool-run-all` orchestrates full pipeline
- [ ] Tracker logs calibration runs correctly
- [ ] Test brain processes start-to-finish without errors
- [ ] napari plugin imports work with new modules

### Tier 2 Complete
- [ ] All utility commands callable via CLI
- [ ] No import conflicts between utils and pipeline
- [ ] Core pipeline still works
- [ ] At least 2 utilities tested end-to-end

### Tier 3 Complete
- [ ] All analysis modules integrated
- [ ] Brain-specific scripts archived
- [ ] No unused imports across package
- [ ] Full test suite passes
- [ ] Deprecation warnings in place for old scripts

---

## Future Considerations

### Beyond This Refactor

1. **Unit tests**: Add tests for each module (currently minimal)
2. **Documentation**: Auto-generate API docs from docstrings
3. **Configuration file**: Support `.braintools.yaml` for project-level settings
4. **Logging**: Unified logging across all CLI commands
5. **Progress tracking**: Live progress reporting during pipeline runs
6. **Docker support**: Package as Docker image with CUDA support
7. **Configuration profiles**: Save/load parameter sets across runs

### Related Refactors

- **napari plugin**: Could eventually use `braintools.cli` directly
- **Batch processing**: Could be further optimized with parallel execution
- **Testing infrastructure**: Create comprehensive test suite with fixtures

---

## Notes & Warnings

### Important Reminders

1. **Do not rush**: Test thoroughly between tiers
2. **Keep napari plugin separate**: Don't try to merge it into braintools
3. **Backward compatibility**: Users may have scripts importing from old locations
4. **Path handling**: Windows paths need careful handling in installed package
5. **Conda environment**: Ensure migration doesn't break environment-based imports

### Known Challenges

| Challenge | Mitigation |
|-----------|-----------|
| Windows path separators | Use `pathlib.Path` consistently |
| Config auto-detection | Add explicit `--base-path` option |
| Tracker CSV location | Environment variable with fallback |
| Script-specific logic | Extract to reusable functions in utils |
| Development workflow | Keep scripts in util_Scripts during transition |

---

## Contact & Questions

This document was created as a planning document for the SCI-Connectome project. For questions or updates:

1. Check `MIGRATION_STATUS.md` for current progress
2. Review PRs for tier-specific changes
3. Refer to individual module claude.md files for implementation details

---

**Last Updated**: 2025-01-26
**Status**: Ready for Tier 1 Implementation
