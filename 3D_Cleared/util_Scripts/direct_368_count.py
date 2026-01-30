#!/usr/bin/env python3
"""Direct counting for brain 368 - minimal script."""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from config import BRAINS_ROOT, DATA_SUMMARY_DIR, REGION_COUNTS_CSV, parse_brain_name

print("Starting direct count for brain 368...")
print(f"Working at: {datetime.now()}")

brain_path = BRAINS_ROOT / "368_CNT_03_08" / "368_CNT_03_08_1p625x_z4"
cells_xml = brain_path / "5_Classified_Cells" / "cells.xml"
reg_path = brain_path / "3_Registered_Atlas"
output_path = brain_path / "6_Region_Analysis"

print(f"Brain path: {brain_path}")
print(f"Cells XML exists: {cells_xml.exists()}")
print(f"Registration exists: {reg_path.exists()}")

if not cells_xml.exists():
    print("ERROR: cells.xml not found!")
    sys.exit(1)

if not reg_path.exists():
    print("ERROR: Registration folder not found!")
    sys.exit(1)

# Count cells from XML
import xml.etree.ElementTree as ET
print("Parsing cells.xml...")
tree = ET.parse(str(cells_xml))
markers = tree.findall('.//Marker')
n_cells = len(markers)
print(f"Total cells in XML: {n_cells}")

# Load brainglobe
print("Loading BrainGlobe libraries...")
try:
    from brainglobe_atlasapi import BrainGlobeAtlas
    from brainglobe_utils.brainmapper.analysis import summarise_points_by_atlas_region
    from brainglobe_utils.brainreg.transform import (
        transform_points_from_raw_to_downsampled_space,
        transform_points_from_downsampled_to_atlas_space,
    )
    import brainglobe_space as bgs
    import numpy as np
    import tifffile
    print("BrainGlobe libraries loaded successfully")
except ImportError as e:
    print(f"ERROR importing brainglobe: {e}")
    sys.exit(1)

# Load atlas
print("Loading atlas...")
atlas = BrainGlobeAtlas("allen_mouse_10um")

# Load registration metadata
brainreg_json = reg_path / "brainreg.json"
with open(brainreg_json) as f:
    reg_meta = json.load(f)

orientation = reg_meta.get('orientation', 'iar')
voxel_sizes = [float(v) for v in reg_meta.get('voxel_sizes', ['4.0', '4.0', '4.0'])]
print(f"Orientation: {orientation}, Voxel sizes: {voxel_sizes}")

# Extract cell coordinates from XML
print("Extracting cell coordinates...")
cells_coords = []
for marker in markers:
    x = int(marker.find('MarkerX').text)
    y = int(marker.find('MarkerY').text)
    z = int(marker.find('MarkerZ').text)
    cells_coords.append([z, y, x])  # ZYX order

cells_array = np.array(cells_coords, dtype=np.float64)
print(f"Extracted {len(cells_array)} cell coordinates")

# Load downsampled shape
downsampled_path = reg_path / "downsampled.tiff"
with tifffile.TiffFile(str(downsampled_path)) as tiff:
    downsampled_shape = tiff.series[0].shape
print(f"Downsampled shape: {downsampled_shape}")

# Create spaces
atlas_resolution = atlas.resolution[0]
target_space = bgs.AnatomicalSpace(
    atlas.space.origin,
    shape=downsampled_shape,
    resolution=[atlas_resolution] * 3,
)

source_space = bgs.AnatomicalSpace(
    orientation,
    shape=downsampled_shape,
    resolution=voxel_sizes,
)

print(f"Source space: {source_space.origin}, Target space: {target_space.origin}")

# Transform coordinates
print("Transforming coordinates to downsampled space...")
downsampled_coords = transform_points_from_raw_to_downsampled_space(
    cells_array,
    target_space,
    source_space,
    voxel_sizes,
)
print(f"Transformed to downsampled: {len(downsampled_coords)} points")

print("Transforming coordinates to atlas space...")
atlas_coords = transform_points_from_downsampled_to_atlas_space(
    downsampled_coords,
    str(reg_path),
)
print(f"Transformed to atlas: {len(atlas_coords)} points")

# Count by region
print("Counting by atlas region...")
region_counts = summarise_points_by_atlas_region(atlas_coords, atlas)

# Build output
output_path.mkdir(parents=True, exist_ok=True)
output_csv = output_path / "cell_counts_by_region.csv"

# Write counts CSV
print(f"Writing counts to {output_csv}...")
import csv
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['region', 'acronym', 'count'])
    for region_name, count in sorted(region_counts.items(), key=lambda x: -x[1]):
        # Get acronym
        try:
            acronym = atlas.structures[region_name]['acronym']
        except:
            acronym = region_name
        writer.writerow([region_name, acronym, count])

print(f"DONE! Total cells: {n_cells}, Regions: {len(region_counts)}")
print(f"Output: {output_csv}")

# Update region_counts.csv
print("\nUpdating region_counts.csv...")
# This would be more complex - just print for now
top_regions = sorted(region_counts.items(), key=lambda x: -x[1])[:10]
print("Top 10 regions:")
for region, count in top_regions:
    print(f"  {region}: {count}")
