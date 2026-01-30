"""
elife_region_mapping.py

Complete mapping from Allen Mouse Brain Atlas regions to eLife summary groups.

The eLife paper (Wang et al., 2022) groups 69 individual brain regions into 25
"summary regions" based on functional/anatomical criteria. This module provides:

1. ELIFE_GROUPS: The 25 summary groups with their constituent Allen regions
2. get_elife_group(): Map any Allen region to its eLife group
3. aggregate_to_elife(): Convert full Allen counts to eLife summary
4. expand_elife_group(): Show what Allen regions are in a group

IMPORTANT: The eLife groupings are FUNCTIONAL categories, not anatomical structures.
For example, "Cerebellospinal Nuclei" means "cerebellar nuclei that project to
spinal cord" - it's not an actual brain region.

Reference:
Wang et al. (2022) eLife. DOI: 10.7554/eLife.76254
"Brain-wide analysis of the supraspinal connectome reveals anatomical
correlates to functional recovery after spinal injury"
"""

from typing import Dict, List, Tuple, Optional, Literal
import re
from enum import Enum

# =============================================================================
# TRACING TYPE - Determines which regions are suspicious
# =============================================================================

class TracingType(Enum):
    """Type of neural tracing experiment."""
    DESCENDING = "descending"  # Retrograde from spinal cord - labels neurons projecting TO cord
    ASCENDING = "ascending"    # ATLAS tracing - labels neurons receiving input FROM cord
    UNKNOWN = "unknown"        # No region filtering applied


# =============================================================================
# SUSPICIOUS REGIONS FOR DESCENDING (RETROGRADE SPINAL) TRACING
# =============================================================================
# These regions should NOT contain cervical-projecting neurons. High counts here
# likely indicate:
# - Meningeal/peripheral false positives (surface structures)
# - Autofluorescence or channel bleed-through
# - Detection artifacts (oversplitting, noise)
#
# Based on analysis of CNT brain data vs eLife cervical reference (2026-01)
#
# IMPORTANT: This is ONLY for retrograde tracing from spinal cord!
# For ATLAS (ascending) tracing, these regions may be completely legitimate.
# See ASCENDING_SUSPICIOUS_REGIONS for ascending tracing criteria.

DESCENDING_SUSPICIOUS_REGIONS = {
    # Category 1: Cerebellar cortex (only deep nuclei project spinally)
    "cerebellar_cortex": {
        "description": "Cerebellar cortex - no spinal projections (only deep nuclei do)",
        "acronyms": [
            "ANcr1", "ANcr2",  # Ansiform lobule crus 1/2
            "NOD",             # Nodulus
            "UVU",             # Uvula
            "SIM",             # Simple lobule
            "CUL4, 5",         # Culmen lobules 4/5
            "PRM",             # Paramedian lobule
            "PFL",             # Paraflocculus
            "FL",              # Flocculus
            "CENT2", "CENT3",  # Central lobule
            "LING",            # Lingula
            "DEC",             # Declive
            "FOTU",            # Folium-tuber vermis
            "PYR",             # Pyramid
            "COPY",            # Copula pyramidis
        ],
    },

    # Category 2: White matter (should have 0 cell bodies)
    "white_matter": {
        "description": "White matter tracts - should contain no cell bodies",
        "acronyms": [
            "arb",             # Arbor vitae (cerebellar white matter)
            "cbc",             # Cerebellar commissure
            "scp", "mcp", "icp",  # Cerebellar peduncles
            "py", "pyd",       # Pyramidal tract/decussation
            "ml", "mlf",       # Medial lemniscus/longitudinal fasciculus
            "fiber tracts",    # Generic fiber tracts
            "or",              # Optic radiation
            "cc", "ec", "int", # Corpus callosum, external capsule, internal capsule
            "fx", "fi",        # Fornix, fimbria
            "st", "sm",        # Stria terminalis, stria medullaris
            "opt", "och",      # Optic tract, optic chiasm
            "cpd",             # Cerebral peduncle
            "em",              # External medullary lamina
        ],
    },

    # Category 3: Olfactory regions (no spinal projections)
    "olfactory": {
        "description": "Olfactory system - no spinal projections",
        "acronyms": [
            "MOB",             # Main olfactory bulb
            "AOB", "AOBgl", "AOBgr", "AOBmi",  # Accessory olfactory bulb
            "OT",              # Olfactory tubercle
            "PIR",             # Piriform area
            "COAa", "COApl", "COApm",  # Cortical amygdalar area
            "NLOT1", "NLOT2", "NLOT3",  # Nucleus of lateral olfactory tract
            "AON",             # Anterior olfactory nucleus
            "TTd", "TTv",      # Taenia tecta
        ],
    },

    # Category 4: Cortical layers 1-3 (only L5/6 have CST neurons)
    "cortical_superficial": {
        "description": "Cortical layers 1-3 - no corticospinal neurons (L5/6 only)",
        "acronyms": [
            # Motor cortex
            "MOp1", "MOp2/3",
            "MOs1", "MOs2/3",
            # Somatosensory
            "SSp-ul1", "SSp-ul2/3",
            "SSp-ll1", "SSp-ll2/3",
            "SSp-tr1", "SSp-tr2/3",
            "SSp-m1", "SSp-m2/3",
            "SSp-n1", "SSp-n2/3",
            "SSp-bfd1", "SSp-bfd2/3", "SSp-bfd4",
            "SSs1", "SSs2/3", "SSs4",
            # Other cortical
            "ACAd1", "ACAd2/3",
            "ACAv1", "ACAv2/3",
            # Visual, auditory, etc. (no spinal projections at all)
            "VISp1", "VISp2/3", "VISp4",
            "AUDp1", "AUDp2/3", "AUDp4",
        ],
    },

    # Category 5: Other unlikely regions
    "other_unlikely": {
        "description": "Other regions unlikely to project to cervical cord",
        "acronyms": [
            "VTA",             # Ventral tegmental area (dopaminergic, not spinal)
            "SNc", "SNr",      # Substantia nigra (basal ganglia circuit)
            "GPe", "GPi",      # Globus pallidus (basal ganglia)
            "STR", "CP",       # Striatum, caudoputamen (basal ganglia)
            "ACB",             # Nucleus accumbens
            "CA1", "CA2", "CA3",  # Hippocampus
            "DG-mo", "DG-po", "DG-sg",  # Dentate gyrus
            "ENTl1", "ENTl2", "ENTl3", "ENTl5", "ENTl6a",  # Entorhinal cortex
            "ENTm1", "ENTm2", "ENTm3", "ENTm5", "ENTm6",
        ],
    },
}

# =============================================================================
# SUSPICIOUS REGIONS FOR ASCENDING (ATLAS) TRACING
# =============================================================================
# Placeholder for future ATLAS tracing support.
# ATLAS traces ascending connections (afferent pathways TO the brain).
# Expected regions would include sensory relay nuclei, cerebellar inputs, etc.
#
# TODO: Define suspicious regions for ascending tracing based on experimental data

ASCENDING_SUSPICIOUS_REGIONS = {
    # Placeholder - to be defined based on ATLAS experimental expectations
    # For now, no regions are flagged as suspicious for ascending tracing
}

# Backwards-compatible alias (defaults to descending/retrograde tracing)
SUSPICIOUS_REGIONS = DESCENDING_SUSPICIOUS_REGIONS

# Build flat set of all suspicious acronyms for descending tracing
_ALL_SUSPICIOUS_DESCENDING = set()
for category, data in DESCENDING_SUSPICIOUS_REGIONS.items():
    _ALL_SUSPICIOUS_DESCENDING.update(data["acronyms"])

_ALL_SUSPICIOUS_ASCENDING = set()
for category, data in ASCENDING_SUSPICIOUS_REGIONS.items():
    _ALL_SUSPICIOUS_ASCENDING.update(data["acronyms"])

# Default (backwards compatible)
_ALL_SUSPICIOUS = _ALL_SUSPICIOUS_DESCENDING


def is_suspicious_region(
    acronym: str,
    tracing_type: TracingType = TracingType.DESCENDING
) -> Optional[str]:
    """
    Check if a region is suspicious based on tracing type.

    Args:
        acronym: Allen atlas region acronym
        tracing_type: Type of tracing experiment (DESCENDING for retrograde spinal,
                      ASCENDING for ATLAS tracing). Default is DESCENDING.

    Returns:
        Category name if suspicious, None if legitimate

    Notes:
        - For DESCENDING (retrograde spinal) tracing: flags regions that don't
          project to spinal cord (cerebellar cortex, olfactory, white matter, etc.)
        - For ASCENDING (ATLAS) tracing: currently no filtering (placeholder)
        - For UNKNOWN: no filtering applied
    """
    # No filtering for unknown or ascending (not yet defined) tracing types
    if tracing_type == TracingType.UNKNOWN:
        return None
    if tracing_type == TracingType.ASCENDING:
        # TODO: Implement ATLAS-specific filtering when criteria are defined
        return None

    # DESCENDING tracing (retrograde from spinal cord)
    suspicious_set = _ALL_SUSPICIOUS_DESCENDING
    suspicious_dict = DESCENDING_SUSPICIOUS_REGIONS

    # Direct check
    if acronym in suspicious_set:
        for cat, data in suspicious_dict.items():
            if acronym in data["acronyms"]:
                return cat

    # Check for patterns (e.g., any layer 1-3 cortical region)
    if re.match(r'.*[123](/[123])?$', acronym):
        # Could be layer 1, 2, 3, or 2/3
        if any(x in acronym for x in ['MOp', 'MOs', 'SSp', 'SSs', 'ACA', 'VIS', 'AUD', 'RSP', 'ORB']):
            return "cortical_superficial"

    # Check for cerebellar lobule patterns
    if re.match(r'^(AN|SIM|NOD|UVU|PRM|CUL|CENT|LING|DEC|FOTU|PYR|COPY|PFL|FL)', acronym):
        return "cerebellar_cortex"

    return None


def get_suspicious_count(
    allen_counts: Dict[str, int],
    tracing_type: TracingType = TracingType.DESCENDING
) -> Dict[str, dict]:
    """
    Categorize counts by suspicious region type.

    Args:
        allen_counts: Dict mapping Allen acronym -> count
        tracing_type: Type of tracing experiment. Default is DESCENDING (retrograde spinal).
                      Use ASCENDING for ATLAS tracing (currently no filtering).

    Returns dict with:
    - "legitimate_elife": count in eLife-mapped regions
    - "suspicious": dict by category with counts
    - "unmapped_legitimate": count in unmapped but not suspicious regions
    - "tracing_type": the tracing type used for filtering
    """
    # Select appropriate suspicious regions dict based on tracing type
    if tracing_type == TracingType.DESCENDING:
        suspicious_dict = DESCENDING_SUSPICIOUS_REGIONS
    elif tracing_type == TracingType.ASCENDING:
        suspicious_dict = ASCENDING_SUSPICIOUS_REGIONS
    else:
        suspicious_dict = {}

    result = {
        "legitimate_elife": 0,
        "suspicious": {cat: {"count": 0, "regions": {}} for cat in suspicious_dict},
        "unmapped_legitimate": 0,
        "unmapped_suspicious": 0,
        "tracing_type": tracing_type.value,
    }

    for acronym, count in allen_counts.items():
        elife_group = get_elife_group(acronym)
        suspicious_cat = is_suspicious_region(acronym, tracing_type)

        if elife_group and elife_group != "[Unmapped]":
            result["legitimate_elife"] += count
        elif suspicious_cat:
            result["suspicious"][suspicious_cat]["count"] += count
            result["suspicious"][suspicious_cat]["regions"][acronym] = count
            result["unmapped_suspicious"] += count
        else:
            result["unmapped_legitimate"] += count

    # Calculate totals
    result["total_suspicious"] = sum(d["count"] for d in result["suspicious"].values())
    result["total_legitimate"] = result["legitimate_elife"] + result["unmapped_legitimate"]
    result["total"] = result["total_suspicious"] + result["total_legitimate"]

    return result


# =============================================================================
# ELIFE SUMMARY GROUPS (25 total)
# =============================================================================
# Each group contains: (group_id, description, [list of Allen atlas acronyms])
# The acronyms match the Allen Mouse Brain Atlas (CCFv3)

ELIFE_GROUPS = {
    # Group 1: Solitariospinal Area
    # Regions around solitary tract that project to spinal cord
    "Solitariospinal Area": {
        "id": 1,
        "description": "Nuclei around solitary tract with spinal projections",
        "acronyms": ["DMX", "CU", "GR", "NTS", "MY"],
        "full_names": [
            "Dorsal motor nucleus of the vagus nerve",
            "Cuneate nucleus",
            "Gracile nucleus",
            "Nucleus of the solitary tract",
            "Medulla",
        ],
    },

    # Group 2: Medullary Reticular Nuclei
    "Medullary Reticular Nuclei": {
        "id": 2,
        "description": "Reticular formation in medulla (dorsal and ventral)",
        "acronyms": ["MDRNd", "MDRNv"],
        "full_names": [
            "Medullary reticular nucleus, dorsal part",
            "Medullary reticular nucleus, ventral part",
        ],
    },

    # Group 3: Magnocellular Reticular Nucleus
    "Magnocellular Reticular Nucleus": {
        "id": 3,
        "description": "Large-celled reticular nucleus",
        "acronyms": ["MARN"],
        "full_names": ["Magnocellular reticular nucleus"],
    },

    # Group 4: Gigantocellular Reticular Nucleus
    "Gigantocellular Reticular Nucleus": {
        "id": 4,
        "description": "Giant-celled reticular nucleus - major reticulospinal source",
        "acronyms": ["GRN"],
        "full_names": ["Gigantocellular reticular nucleus"],
    },

    # Group 5: Lateral Reticular Nuclei
    "Lateral Reticular Nuclei": {
        "id": 5,
        "description": "Lateral reticular formation nuclei",
        "acronyms": ["LRNm", "LRNp", "PGRNl", "IRN", "PARN"],
        "full_names": [
            "Lateral reticular nucleus, magnocellular part",
            "Lateral reticular nucleus, parvicellular part",
            "Paragigantocellular reticular nucleus, lateral part",
            "Intermediate reticular nucleus",
            "Parvicellular reticular nucleus",
        ],
    },

    # Group 6: Dorsal Reticular Nucleus (Paragigantocellular dorsal)
    "Dorsal Reticular Nucleus": {
        "id": 6,
        "description": "Dorsal paragigantocellular region",
        "acronyms": ["PGRNd"],
        "full_names": ["Paragigantocellular reticular nucleus, dorsal part"],
    },

    # Group 7: Raphe Nuclei
    "Raphe Nuclei": {
        "id": 7,
        "description": "Midline raphe nuclei (serotonergic)",
        "acronyms": ["RM", "RO", "RPO", "RPA", "DR", "CS", "RL"],
        "full_names": [
            "Nucleus raphe magnus",
            "Nucleus raphe obscurus",
            "Nucleus raphe pontis",
            "Nucleus raphe pallidus",
            "Dorsal nucleus raphe",
            "Superior central nucleus raphe",
            "Rostral linear nucleus raphe",
        ],
    },

    # Group 8: Perihypoglossal Area
    "Perihypoglossal Area": {
        "id": 8,
        "description": "Nuclei around hypoglossal nucleus",
        "acronyms": ["XII", "NR", "PRP"],
        "full_names": [
            "Hypoglossal nucleus",
            "Nucleus of Roller",
            "Nucleus prepositus",
        ],
    },

    # Group 9: Medullary Trigeminal Area
    "Medullary Trigeminal Area": {
        "id": 9,
        "description": "Spinal trigeminal nucleus (medullary portions)",
        "acronyms": ["SPVI", "SPVC", "SPVO"],
        "full_names": [
            "Spinal nucleus of the trigeminal, interpolar part",
            "Spinal nucleus of the trigeminal, caudal part",
            "Spinal nucleus of the trigeminal, oral part",
        ],
    },

    # Group 10: Vestibular Nuclei
    "Vestibular Nuclei": {
        "id": 10,
        "description": "Vestibular nuclear complex",
        "acronyms": ["SPIV", "MV", "LAV", "SUV"],
        "full_names": [
            "Spinal vestibular nucleus",
            "Medial vestibular nucleus",
            "Lateral vestibular nucleus",
            "Superior vestibular nucleus",
        ],
    },

    # Group 11: Pontine Reticular Nuclei
    "Pontine Reticular Nuclei": {
        "id": 11,
        "description": "Pontine reticular formation",
        "acronyms": ["PRNc", "PRNr"],
        "full_names": [
            "Pontine reticular nucleus, caudal part",
            "Pontine reticular nucleus",
        ],
    },

    # Group 12: Superior Olivary Complex
    "Superior Olivary Complex": {
        "id": 12,
        "description": "Superior olivary nuclei (auditory)",
        "acronyms": ["SOCl", "SOCm", "POR"],
        "full_names": [
            "Superior olivary complex, lateral part",
            "Superior olivary complex, medial part",
            "Superior olivary complex, periolivary region",
        ],
    },

    # Group 13: Pontine Trigeminal Area
    "Pontine Trigeminal Area": {
        "id": 13,
        "description": "Trigeminal-related pontine nuclei",
        "acronyms": ["P5", "SUT", "PSV", "V"],
        "full_names": [
            "Peritrigeminal zone",
            "Supratrigeminal nucleus",
            "Principal sensory nucleus of the trigeminal",
            "Motor nucleus of trigeminal",
        ],
    },

    # Group 14: Pontine Central Gray Area
    "Pontine Central Gray Area": {
        "id": 14,
        "description": "Central gray and tegmental nuclei of pons",
        "acronyms": ["PCG", "SLD", "LDT", "SLC", "B", "P", "LC", "KF"],
        "full_names": [
            "Pontine central gray",
            "Sublaterodorsal nucleus",
            "Laterodorsal tegmental nucleus",
            "Subceruleus nucleus",
            "Barrington's nucleus",
            "Pons",
            "Locus ceruleus",
            "Koelliker-Fuse subnucleus",
        ],
    },

    # Group 15: Parabrachial / Pedunculopontine
    "Parabrachial / Pedunculopontine": {
        "id": 15,
        "description": "Parabrachial and pedunculopontine tegmental nuclei",
        "acronyms": ["PB", "PPN"],
        "full_names": [
            "Parabrachial nucleus",
            "Pedunculopontine nucleus",
        ],
    },

    # Group 16: Cerebellospinal Nuclei
    # NOTE: This is a FUNCTIONAL grouping - cerebellar nuclei that project to spinal cord
    "Cerebellospinal Nuclei": {
        "id": 16,
        "description": "Deep cerebellar nuclei with spinal projections (functional group)",
        "acronyms": ["FN", "IP", "DN"],
        "full_names": [
            "Fastigial nucleus",
            "Interposed nucleus",
            "Dentate nucleus",
        ],
    },

    # Group 17: Red Nucleus
    "Red Nucleus": {
        "id": 17,
        "description": "Red nucleus - major rubrospinal source",
        "acronyms": ["RN"],
        "full_names": ["Red nucleus"],
    },

    # Group 18: Midbrain Midline Nuclei
    "Midbrain Midline Nuclei": {
        "id": 18,
        "description": "Midline nuclei of midbrain tegmentum",
        "acronyms": ["EW", "INC", "ND", "MB", "Su3", "MA3"],
        "full_names": [
            "Edinger-Westphal nucleus",
            "Interstitial nucleus of Cajal",
            "Nucleus of Darkschewitsch",
            "Midbrain",
            "Supraoculomotor periaqueductal gray",
            "Medial accessory oculomotor nucleus",
        ],
    },

    # Group 19: Midbrain Reticular Nuclei
    "Midbrain Reticular Nuclei": {
        "id": 19,
        "description": "Midbrain reticular formation",
        "acronyms": ["MRN", "RR", "CUN"],
        "full_names": [
            "Midbrain reticular nucleus",
            "Midbrain reticular nucleus, retrorubral area",
            "Cuneiform nucleus",
        ],
    },

    # Group 20: Periaqueductal Gray
    "Periaqueductal Gray": {
        "id": 20,
        "description": "Periaqueductal gray matter",
        "acronyms": ["PAG"],
        "full_names": ["Periaqueductal gray"],
    },

    # Group 21: Hypothalamic Periventricular Zone
    "Hypothalamic Periventricular Zone": {
        "id": 21,
        "description": "Periventricular hypothalamic nuclei",
        "acronyms": ["PVH", "PVHd", "PVi", "SBPV"],
        "full_names": [
            "Paraventricular hypothalamic nucleus",
            "Paraventricular hypothalamic nucleus, descending division",
            "Periventricular hypothalamic nucleus, intermediate part",
            "Subparaventricular zone",
        ],
    },

    # Group 22: Hypothalamic Lateral Area
    "Hypothalamic Lateral Area": {
        "id": 22,
        "description": "Lateral hypothalamic regions",
        "acronyms": ["DMH", "HY", "LHA", "ZI", "PeF", "TU", "PSTN", "STN"],
        "full_names": [
            "Dorsomedial nucleus of the hypothalamus",
            "Hypothalamus",
            "Lateral hypothalamic area",
            "Zona incerta",
            "Perifornical nucleus",
            "Tuberal nucleus",
            "Parasubthalamic nucleus",
            "Subthalamic nucleus",
        ],
    },

    # Group 23: Hypothalamic Medial Area
    "Hypothalamic Medial Area": {
        "id": 23,
        "description": "Medial hypothalamic nuclei",
        "acronyms": ["AHN", "PH", "VMH"],
        "full_names": [
            "Anterior hypothalamic nucleus",
            "Posterior hypothalamic nucleus",
            "Ventromedial hypothalamic nucleus",
        ],
    },

    # Group 24: Thalamus
    "Thalamus": {
        "id": 24,
        "description": "Thalamic nuclei with descending projections",
        "acronyms": ["PR", "SPFm", "SPFp", "TH", "VM", "RE", "PF", "VAL", "MD"],
        "full_names": [
            "Perireunensis nucleus",
            "Subparafascicular nucleus, magnocellular part",
            "Subparafascicular nucleus, parvicellular part",
            "Thalamus",
            "Ventral medial nucleus of the thalamus",
            "Nucleus of reuniens",
            "Parafascicular nucleus",
            "Ventral anterior-lateral complex of the thalamus",
            "Mediodorsal nucleus of thalamus",
        ],
    },

    # Group 25: Corticospinal
    # NOTE: This includes motor and somatosensory cortex layers 5 and 6
    "Corticospinal": {
        "id": 25,
        "description": "Cortical layers 5/6 with corticospinal projections",
        "acronyms": [
            # Primary motor
            "MOp5", "MOp6a", "MOp6b",
            # Secondary motor
            "MOs5", "MOs6a", "MOs6b",
            # Primary somatosensory - all body regions
            "SSp-ul5", "SSp-ul6a",  # upper limb
            "SSp-ll5", "SSp-ll6a",  # lower limb
            "SSp-tr5", "SSp-tr6a",  # trunk
            "SSp-m5", "SSp-m6a",    # mouth
            "SSp-n5", "SSp-n6a",    # nose
            "SSp-bfd5", "SSp-bfd6a", # barrel field
            "SSp-un5", "SSp-un6a",  # unassigned
            # Supplemental somatosensory
            "SSs5", "SSs6a",
            # Anterior cingulate (has some CST neurons)
            "ACAd5", "ACAd6a", "ACAv5", "ACAv6a",
        ],
        "full_names": [
            "Primary motor area, Layer 5",
            "Primary motor area, Layer 6a",
            "Primary motor area, Layer 6b",
            "Secondary motor area, layer 5",
            "Secondary motor area, layer 6a",
            "Secondary motor area, layer 6b",
            # ... (full names for all)
        ],
    },

    # Group 30: Unused (regions eLife explicitly excluded)
    "Unused": {
        "id": 30,
        "description": "Regions excluded from eLife analysis",
        "acronyms": ["VII", "LC", "CEAm", "SI"],
        "full_names": [
            "Facial motor nucleus",
            "Locus ceruleus",  # Note: LC appears in both 14 and 30 in eLife data
            "Central amygdalar nucleus, medial part",
            "Substantia innominata",
        ],
    },
}

# Build reverse lookup: acronym -> eLife group name
_ACRONYM_TO_GROUP = {}
for group_name, group_data in ELIFE_GROUPS.items():
    for acronym in group_data["acronyms"]:
        _ACRONYM_TO_GROUP[acronym] = group_name


def get_elife_group(acronym: str) -> Optional[str]:
    """
    Get the eLife summary group for an Allen atlas acronym.

    Args:
        acronym: Allen atlas region acronym (e.g., "GRN", "MOp5")

    Returns:
        eLife group name or None if not mapped
    """
    # Direct lookup
    if acronym in _ACRONYM_TO_GROUP:
        return _ACRONYM_TO_GROUP[acronym]

    # Try without layer suffix for cortical regions
    # e.g., "SSp-ul5" -> check "SSp-ul"
    base = re.sub(r'[0-9/]+[ab]?$', '', acronym)
    if base in _ACRONYM_TO_GROUP:
        return _ACRONYM_TO_GROUP[base]

    # Check if it's a cortical layer 5 or 6 region
    if re.search(r'[56][ab]?$', acronym):
        # Likely corticospinal if it's a layer 5/6 sensorimotor region
        if any(x in acronym for x in ['MOp', 'MOs', 'SSp', 'SSs', 'ACA']):
            return "Corticospinal"

    return None


def aggregate_to_elife(allen_counts: Dict[str, int]) -> Dict[str, dict]:
    """
    Aggregate Allen atlas counts to eLife summary groups.

    Args:
        allen_counts: Dict mapping Allen acronym -> count

    Returns:
        Dict mapping eLife group -> {
            "count": total count,
            "constituents": {acronym: count, ...}
        }
    """
    result = {}
    unmapped = {}

    for acronym, count in allen_counts.items():
        group = get_elife_group(acronym)

        if group:
            if group not in result:
                result[group] = {"count": 0, "constituents": {}}
            result[group]["count"] += count
            result[group]["constituents"][acronym] = count
        else:
            unmapped[acronym] = count

    # Add unmapped as special group
    if unmapped:
        result["[Unmapped]"] = {
            "count": sum(unmapped.values()),
            "constituents": unmapped
        }

    return result


def expand_elife_group(group_name: str) -> List[str]:
    """Get list of Allen acronyms in an eLife group."""
    if group_name in ELIFE_GROUPS:
        return ELIFE_GROUPS[group_name]["acronyms"]
    return []


def get_group_description(group_name: str) -> str:
    """Get description for an eLife group."""
    if group_name in ELIFE_GROUPS:
        return ELIFE_GROUPS[group_name]["description"]
    return ""


# =============================================================================
# CONVENIENCE FUNCTIONS FOR OUTPUT
# =============================================================================

def format_elife_summary(allen_counts: Dict[str, int], show_constituents: bool = False) -> str:
    """
    Format Allen counts as eLife-style summary text.

    Args:
        allen_counts: Dict mapping Allen acronym -> count
        show_constituents: If True, show individual regions under each group

    Returns:
        Formatted string
    """
    aggregated = aggregate_to_elife(allen_counts)

    lines = ["=" * 70]
    lines.append("ELIFE-STYLE SUMMARY (grouped by Wang et al. 2022 categories)")
    lines.append("=" * 70)

    # Sort by count descending
    sorted_groups = sorted(
        aggregated.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )

    total = 0
    for group_name, data in sorted_groups:
        count = data["count"]
        total += count
        lines.append(f"{group_name:<45} {count:>8,}")

        if show_constituents and group_name != "[Unmapped]":
            for acronym, acr_count in sorted(
                data["constituents"].items(),
                key=lambda x: -x[1]
            ):
                lines.append(f"    {acronym:<41} {acr_count:>8,}")

    lines.append("-" * 70)
    lines.append(f"{'TOTAL':<45} {total:>8,}")

    return "\n".join(lines)


def save_dual_output(
    allen_counts: Dict[str, int],
    output_dir,
    brain_name: str = ""
) -> Tuple[str, str]:
    """
    Save both full Allen detail and eLife summary CSVs.

    Args:
        allen_counts: Dict mapping Allen acronym -> count
        output_dir: Directory to save files
        brain_name: Optional brain name for filename prefix

    Returns:
        Tuple of (detail_path, summary_path)
    """
    from pathlib import Path
    import csv

    output_dir = Path(output_dir)
    prefix = f"{brain_name}_" if brain_name else ""

    # 1. Full detail CSV (unchanged - this is what we always save)
    detail_path = output_dir / f"{prefix}cell_counts_by_region.csv"

    # 2. eLife-grouped summary
    summary_path = output_dir / f"{prefix}cell_counts_elife_grouped.csv"
    aggregated = aggregate_to_elife(allen_counts)

    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "elife_group",
            "group_id",
            "cell_count",
            "constituent_regions",
            "description"
        ])

        for group_name in sorted(ELIFE_GROUPS.keys(), key=lambda x: ELIFE_GROUPS[x]["id"]):
            if group_name in aggregated:
                data = aggregated[group_name]
                constituents = "; ".join(
                    f"{k}={v}" for k, v in sorted(
                        data["constituents"].items(),
                        key=lambda x: -x[1]
                    )
                )
                writer.writerow([
                    group_name,
                    ELIFE_GROUPS[group_name]["id"],
                    data["count"],
                    constituents,
                    ELIFE_GROUPS[group_name]["description"],
                ])
            else:
                writer.writerow([
                    group_name,
                    ELIFE_GROUPS[group_name]["id"],
                    0,
                    "",
                    ELIFE_GROUPS[group_name]["description"],
                ])

        # Add unmapped if present
        if "[Unmapped]" in aggregated:
            data = aggregated["[Unmapped]"]
            constituents = "; ".join(
                f"{k}={v}" for k, v in sorted(
                    data["constituents"].items(),
                    key=lambda x: -x[1]
                )[:20]  # Limit to top 20
            )
            writer.writerow([
                "[Unmapped]",
                99,
                data["count"],
                constituents + ("..." if len(data["constituents"]) > 20 else ""),
                "Regions not in eLife mapping",
            ])

    return str(detail_path), str(summary_path)


def format_corrected_summary(
    allen_counts: Dict[str, int],
    tracing_type: TracingType = TracingType.DESCENDING
) -> str:
    """
    Format a summary showing both raw and corrected (excluding suspicious) counts.

    This is the recommended format for quick summaries.

    Args:
        allen_counts: Dict mapping Allen acronym -> count
        tracing_type: Type of tracing experiment. Default is DESCENDING (retrograde spinal).
                      For ATLAS (ASCENDING) tracing, no suspicious filtering is applied.
    """
    suspicious = get_suspicious_count(allen_counts, tracing_type)
    aggregated = aggregate_to_elife(allen_counts)

    lines = ["=" * 80]
    tracing_label = tracing_type.value.upper()
    lines.append(f"CELL COUNT SUMMARY - {tracing_label} TRACING")
    lines.append("=" * 80)
    lines.append("")

    # Totals section
    total = suspicious["total"]
    legitimate = suspicious["total_legitimate"]
    suspicious_total = suspicious["total_suspicious"]

    lines.append("TOTALS:")
    lines.append(f"  Raw total:        {total:>8,} cells")
    lines.append(f"  Legitimate:       {legitimate:>8,} cells ({legitimate/total*100:.1f}%)")

    if tracing_type == TracingType.DESCENDING and suspicious_total > 0:
        lines.append(f"  Suspicious:       {suspicious_total:>8,} cells ({suspicious_total/total*100:.1f}%) <- likely false positives")
    elif tracing_type == TracingType.ASCENDING:
        lines.append("  (No suspicious region filtering for ATLAS/ascending tracing)")

    lines.append("")

    # eLife-mapped regions (the "real" signal)
    lines.append("eLIFE-MAPPED REGIONS (25 groups, likely true signal):")
    lines.append(f"  Total: {suspicious['legitimate_elife']:,} cells")
    lines.append("")

    # Show key recovery regions
    lines.append("  Key recovery regions:")
    key_regions = ["Red Nucleus", "Gigantocellular Reticular Nucleus", "Corticospinal",
                   "Parabrachial / Pedunculopontine", "Pontine Central Gray Area"]
    for region in key_regions:
        if region in aggregated:
            lines.append(f"    {region}: {aggregated[region]['count']:,}")

    lines.append("")

    # Suspicious breakdown (only for descending tracing)
    if tracing_type == TracingType.DESCENDING and suspicious_total > 0:
        lines.append("SUSPICIOUS REGIONS (likely false positives for descending tracing):")
        for cat, data in suspicious["suspicious"].items():
            if data["count"] > 0:
                desc = DESCENDING_SUSPICIOUS_REGIONS[cat]["description"]
                lines.append(f"  {cat}: {data['count']:,} ({desc})")
                # Top regions
                top_regions = sorted(data["regions"].items(), key=lambda x: -x[1])[:5]
                for region, count in top_regions:
                    lines.append(f"      {region}: {count:,}")

        lines.append("")
        lines.append("-" * 80)
        lines.append("RECOMMENDATION: Use 'Legitimate' total for comparisons to published data.")
        lines.append("Suspicious regions are flagged due to: peripheral/meningeal signal,")
        lines.append("cerebellar cortex (no spinal projections), white matter, olfactory regions.")
    elif tracing_type == TracingType.ASCENDING:
        lines.append("-" * 80)
        lines.append("NOTE: This is ATLAS (ascending) tracing data.")
        lines.append("No suspicious region filtering applied - criteria not yet defined.")
        lines.append("Expected regions differ from descending (retrograde spinal) tracing.")

    return "\n".join(lines)


def get_corrected_total(
    allen_counts: Dict[str, int],
    tracing_type: TracingType = TracingType.DESCENDING
) -> int:
    """
    Get the corrected total excluding suspicious regions.

    Use this for quick summaries and comparisons.

    Args:
        allen_counts: Dict mapping Allen acronym -> count
        tracing_type: Type of tracing experiment. Default is DESCENDING (retrograde spinal).
                      For ASCENDING (ATLAS) tracing, returns total (no filtering).
    """
    suspicious = get_suspicious_count(allen_counts, tracing_type)
    return suspicious["total_legitimate"]


# =============================================================================
# MAIN - Demo/test
# =============================================================================

if __name__ == "__main__":
    # Demo with sample data
    sample_counts = {
        "GRN": 3721,
        "MRN": 1525,
        "RN": 1522,
        "PRNr": 1483,
        "MOp5": 1359,
        "SSp-ul5": 1062,
        "MARN": 1018,
        "PAG": 882,
        "PRNc": 821,
        "FN": 149,
        "IP": 90,
        "VTA": 59,  # Not in eLife mapping
    }

    print(format_elife_summary(sample_counts, show_constituents=True))

    print("\n\nExample aggregation:")
    agg = aggregate_to_elife(sample_counts)
    for group, data in sorted(agg.items(), key=lambda x: -x[1]["count"]):
        print(f"{group}: {data['count']} ({list(data['constituents'].keys())})")
