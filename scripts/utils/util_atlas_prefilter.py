#!/usr/bin/env python3
"""
util_atlas_prefilter.py - Backward-compatibility shim.

The canonical version of this module now lives in the mousebrain package:
    from mousebrain.prefilter import prefilter_candidates, save_prefilter_results

This shim re-exports everything so existing scripts that do
    from util_atlas_prefilter import prefilter_candidates
continue to work without changes.
"""

try:
    from mousebrain.prefilter import *  # noqa: F401,F403
    from mousebrain.prefilter import (  # noqa: F401
        prefilter_candidates,
        save_prefilter_results,
        _coords_to_xml,
        find_brain_path,
        find_latest_candidates,
        main,
        SCRIPT_VERSION,
        FOLDER_REGISTRATION,
        FOLDER_DETECTION,
        timestamp,
    )
except ImportError:
    raise ImportError(
        "mousebrain package is required. Install with:\n"
        "  cd Y:\\2_Connectome\\Tissue\\MouseBrain\n"
        "  pip install -e ."
    )

if __name__ == "__main__":
    main()
