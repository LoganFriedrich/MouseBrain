# Architecture Decision Log

This document tracks key technical decisions made during development. Add new decisions at the top.

---

## Template

### ADR-XXX: [Title]
**Date**: YYYY-MM-DD
**Status**: Proposed | Accepted | Deprecated | Superseded
**Context**: What is the issue?
**Decision**: What did we decide?
**Consequences**: What are the trade-offs?

---

## Decisions

### ADR-001: Use ABBA for 2D slice registration
**Date**: 2025-01-07
**Status**: Proposed

**Context**:
Need to register 2D brain slices to a standard atlas. Options considered:
- ABBA (Aligning Big Brains & Atlases)
- brainreg (BrainGlobe)
- Custom elastix-based solution
- QuPath + ABBA

**Decision**:
Use ABBA via napari-abba plugin as primary registration tool.

**Rationale**:
1. ABBA is specifically designed for 2D-to-3D atlas registration
2. Has mature napari integration
3. Supports non-linear deformations
4. Active development and community
5. Can integrate with DeepSlice for faster initial positioning

**Consequences**:
- (+) Well-tested workflow for slice registration
- (+) Interactive refinement in napari
- (-) Requires Java runtime
- (-) Learning curve for ABBA-specific concepts
- (-) May need to extract transforms for custom downstream analysis

---

### ADR-002: Atlas selection (TBD)
**Date**: 2025-01-07
**Status**: Proposed

**Context**:
Need to select reference atlas. Common options:
- Allen Mouse Brain CCF (10um, 25um, 50um)
- Waxholm Rat Atlas
- Kim Unified Atlas (developmental)

**Decision**:
TBD - depends on species and application

**Questions to resolve**:
- What species are you working with?
- What resolution do you need?
- Any specific brain regions of focus?

---

### ADR-003: Cell detection method (TBD)
**Date**: 2025-01-07
**Status**: Proposed

**Context**:
Need to detect and count cells. Options:
- StarDist: Fast, good for round nuclei
- Cellpose: Flexible, handles varied morphologies
- cellfinder: BrainGlobe integrated

**Decision**:
TBD - depends on cell morphology and staining

**Questions to resolve**:
- What staining method (DAPI, antibody, etc.)?
- Cell morphology (round nuclei vs varied shapes)?
- Need for cell type classification?

---

## Pending Decisions

- [ ] ADR-004: Image format and storage strategy
- [ ] ADR-005: Batch processing approach
- [ ] ADR-006: Data export format (CSV, JSON, database)
- [ ] ADR-007: napari plugin architecture (single vs multiple widgets)
