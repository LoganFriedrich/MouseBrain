#!/usr/bin/env python3
"""
elife_region_mapping.py - Backward-compatibility shim.

The canonical version of this module now lives in the mousebrain package:
    from mousebrain.region_mapping import is_suspicious_region, TracingType

This shim re-exports everything so existing scripts that do
    from elife_region_mapping import is_suspicious_region
continue to work without changes.
"""

try:
    from mousebrain.region_mapping import *  # noqa: F401,F403
    from mousebrain.region_mapping import (  # noqa: F401
        TracingType,
        is_suspicious_region,
        get_suspicious_count,
        get_elife_group,
        aggregate_to_elife,
        expand_elife_group,
        get_group_description,
        format_elife_summary,
        save_dual_output,
        format_corrected_summary,
        get_corrected_total,
        ELIFE_GROUPS,
        DESCENDING_SUSPICIOUS_REGIONS,
        ASCENDING_SUSPICIOUS_REGIONS,
        SUSPICIOUS_REGIONS,
    )
except ImportError:
    raise ImportError(
        "mousebrain package is required. Install with:\n"
        "  cd Y:\\2_Connectome\\Tissue\\MouseBrain\n"
        "  pip install -e ."
    )
