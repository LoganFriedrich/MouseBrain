#!/usr/bin/env python3
"""Simple script to run counting on brain 368 with explicit file logging."""
import sys
from pathlib import Path

# Create log file right away
log_path = Path(__file__).parent / "368_run_log.txt"
with open(log_path, 'w') as f:
    f.write("Starting script...\n")

def log(msg):
    with open(log_path, 'a') as f:
        f.write(msg + "\n")
    print(msg)

log("Setting up paths...")
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config import BRAINS_ROOT
    log(f"BRAINS_ROOT: {BRAINS_ROOT}")

    brain_path = BRAINS_ROOT / "368_CNT_03_08" / "368_CNT_03_08_1p625x_z4"
    log(f"Brain path: {brain_path}")
    log(f"Exists: {brain_path.exists()}")

    cells_xml = brain_path / "5_Classified_Cells" / "cells.xml"
    log(f"Cells XML exists: {cells_xml.exists()}")

    reg_path = brain_path / "3_Registered_Atlas"
    log(f"Registration exists: {reg_path.exists()}")

    if cells_xml.exists():
        import xml.etree.ElementTree as ET
        log("Parsing XML...")
        tree = ET.parse(str(cells_xml))
        markers = tree.findall('.//Marker')
        log(f"Found {len(markers)} cells")

    log("DONE")

except Exception as e:
    log(f"ERROR: {e}")
    import traceback
    log(traceback.format_exc())
