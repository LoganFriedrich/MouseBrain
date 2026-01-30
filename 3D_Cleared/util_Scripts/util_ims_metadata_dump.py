#!/usr/bin/env python3
"""
util_ims_metadata_dump.py

Utility: Dump all metadata from an .ims file to see what's actually in there.

Just run it - it will find your .ims files and let you pick one.
"""

import sys
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
from config import BRAINS_ROOT as DEFAULT_BRAINGLOBE_ROOT


def find_ims_files(root_path):
    """Find all .ims files recursively."""
    root_path = Path(root_path)
    ims_files = []
    
    for ims_file in root_path.rglob("*.ims"):
        ims_files.append(ims_file)
    
    return sorted(ims_files)


def decode_ims_value(val):
    """
    Decode an IMS metadata value.
    
    IMS stores values as arrays of single-character bytes like:
        [b'-' b'8' b'4' b'5' b'.' b'4' b'1' b'9']
    
    This function joins them into a proper string: "-845.419"
    """
    if val is None:
        return None
    
    # If it's already a simple type, return it
    if isinstance(val, (int, float)):
        return val
    
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='replace')
    
    if isinstance(val, str):
        return val
    
    # If it's an array/list, try to join the bytes
    try:
        if hasattr(val, '__iter__'):
            # Check if it's an array of single bytes
            chars = []
            for item in val:
                if isinstance(item, bytes):
                    chars.append(item.decode('utf-8', errors='replace'))
                elif isinstance(item, (int, float)):
                    # Could be ASCII codes or actual numbers
                    if 0 <= item <= 127:
                        chars.append(chr(int(item)))
                    else:
                        return val  # Not character data, return as-is
                else:
                    chars.append(str(item))
            
            result = ''.join(chars)
            
            # Try to convert to number if it looks like one
            try:
                if '.' in result:
                    return float(result)
                elif result.lstrip('-').isdigit():
                    return int(result)
            except:
                pass
            
            return result
    except:
        pass
    
    return val


def dump_ims_metadata(filepath):
    """Dump all metadata from an IMS file."""
    import h5py
    
    filepath = Path(filepath)
    output_path = filepath.parent / f"{filepath.stem}_metadata.txt"
    
    print(f"\nReading: {filepath.name}")
    print(f"Output:  {output_path.name}")
    print()
    
    lines = []
    lines.append("=" * 70)
    lines.append(f"IMS METADATA DUMP")
    lines.append(f"File: {filepath.name}")
    lines.append("=" * 70)
    lines.append("")
    
    voxel_info = {}  # Store for console display
    
    try:
        with h5py.File(filepath, 'r') as f:
            
            # Focus on the important stuff first
            lines.append("=" * 70)
            lines.append("VOXEL SIZE INFO")
            lines.append("=" * 70)
            
            if 'DataSetInfo/Image' in f:
                img = f['DataSetInfo/Image']
                
                # Extract and decode key values
                ext_min_x = decode_ims_value(img.attrs.get('ExtMin0'))
                ext_max_x = decode_ims_value(img.attrs.get('ExtMax0'))
                ext_min_y = decode_ims_value(img.attrs.get('ExtMin1'))
                ext_max_y = decode_ims_value(img.attrs.get('ExtMax1'))
                ext_min_z = decode_ims_value(img.attrs.get('ExtMin2'))
                ext_max_z = decode_ims_value(img.attrs.get('ExtMax2'))
                
                size_x = decode_ims_value(img.attrs.get('X'))
                size_y = decode_ims_value(img.attrs.get('Y'))
                size_z = decode_ims_value(img.attrs.get('Z'))
                
                unit = decode_ims_value(img.attrs.get('Unit'))
                
                lines.append(f"  Unit: {unit}")
                lines.append("")
                lines.append(f"  X: ExtMin={ext_min_x}, ExtMax={ext_max_x}, Size={size_x} pixels")
                lines.append(f"  Y: ExtMin={ext_min_y}, ExtMax={ext_max_y}, Size={size_y} pixels")
                lines.append(f"  Z: ExtMin={ext_min_z}, ExtMax={ext_max_z}, Size={size_z} slices")
                lines.append("")
                
                # Calculate voxel sizes
                lines.append("CALCULATED VOXEL SIZES:")
                
                try:
                    if all(v is not None for v in [ext_min_x, ext_max_x, size_x]):
                        ext_min_x = float(ext_min_x)
                        ext_max_x = float(ext_max_x)
                        size_x = float(size_x)
                        voxel_x = (ext_max_x - ext_min_x) / size_x
                        lines.append(f"  X voxel = ({ext_max_x} - {ext_min_x}) / {int(size_x)} = {voxel_x:.4f} {unit}")
                        voxel_info['x'] = voxel_x
                except Exception as e:
                    lines.append(f"  X voxel: could not calculate ({e})")
                
                try:
                    if all(v is not None for v in [ext_min_y, ext_max_y, size_y]):
                        ext_min_y = float(ext_min_y)
                        ext_max_y = float(ext_max_y)
                        size_y = float(size_y)
                        voxel_y = (ext_max_y - ext_min_y) / size_y
                        lines.append(f"  Y voxel = ({ext_max_y} - {ext_min_y}) / {int(size_y)} = {voxel_y:.4f} {unit}")
                        voxel_info['y'] = voxel_y
                except Exception as e:
                    lines.append(f"  Y voxel: could not calculate ({e})")
                
                try:
                    if all(v is not None for v in [ext_min_z, ext_max_z, size_z]):
                        ext_min_z = float(ext_min_z)
                        ext_max_z = float(ext_max_z)
                        size_z = float(size_z)
                        voxel_z = (ext_max_z - ext_min_z) / size_z
                        lines.append(f"  Z voxel = ({ext_max_z} - {ext_min_z}) / {int(size_z)} = {voxel_z:.4f} {unit}")
                        voxel_info['z'] = voxel_z
                except Exception as e:
                    lines.append(f"  Z voxel: could not calculate ({e})")
                
                lines.append("")
                
                # Summary box
                if voxel_info:
                    lines.append("=" * 70)
                    lines.append("SUMMARY - USE THESE VALUES")
                    lines.append("=" * 70)
                    vx = voxel_info.get('x', '?')
                    vy = voxel_info.get('y', '?')
                    vz = voxel_info.get('z', '?')
                    
                    if isinstance(vx, float) and isinstance(vy, float) and isinstance(vz, float):
                        lines.append(f"  Voxel size (X, Y, Z): {vx:.2f} x {vy:.2f} x {vz:.2f} {unit}")
                        lines.append(f"  For brainreg: -v {vz:.2f} {vy:.2f} {vx:.2f}")
                        lines.append(f"  For napari scale: ({vz:.2f}, {vy:.2f}, {vx:.2f})")
                    lines.append("")
            
            # Full metadata tree
            lines.append("=" * 70)
            lines.append("FULL METADATA TREE")
            lines.append("=" * 70)
            
            def dump_group(group, prefix=""):
                """Recursively dump a group."""
                result = []
                
                # Dump attributes
                for key in sorted(group.attrs.keys()):
                    val = decode_ims_value(group.attrs[key])
                    result.append(f"{prefix}{key} = {val}")
                
                # Recurse into subgroups
                for key in sorted(group.keys()):
                    result.append(f"\n{prefix}{key}/")
                    try:
                        child = group[key]
                        if hasattr(child, 'keys'):
                            result.extend(dump_group(child, prefix + "  "))
                    except Exception as e:
                        result.append(f"{prefix}  <error: {e}>")
                
                return result
            
            lines.extend(dump_group(f))
    
    except Exception as e:
        lines.append(f"ERROR: {e}")
        import traceback
        lines.append(traceback.format_exc())
    
    # Print key info to console
    print("=" * 60)
    print("VOXEL SIZES FROM .IMS FILE")
    print("=" * 60)
    
    if voxel_info:
        vx = voxel_info.get('x', '?')
        vy = voxel_info.get('y', '?')
        vz = voxel_info.get('z', '?')
        
        if isinstance(vx, float) and isinstance(vy, float) and isinstance(vz, float):
            print(f"\n  X voxel: {vx:.4f} um")
            print(f"  Y voxel: {vy:.4f} um")
            print(f"  Z voxel: {vz:.4f} um")
            print()
            print(f"  For brainreg:    -v {vz:.2f} {vy:.2f} {vx:.2f}")
            print(f"  For napari:      scale=({vz:.2f}, {vy:.2f}, {vx:.2f})")
            print()
            
            # Sanity check
            print("-" * 60)
            print("SANITY CHECK:")
            if vx < 0.5 or vx > 20:
                print(f"  WARNING: X voxel ({vx:.2f}) seems unusual (expected 1-10 um)")
            if vy < 0.5 or vy > 20:
                print(f"  WARNING: Y voxel ({vy:.2f}) seems unusual (expected 1-10 um)")
            if vz < 0.5 or vz > 20:
                print(f"  WARNING: Z voxel ({vz:.2f}) seems unusual (expected 1-10 um)")
            if abs(vx - vy) > 0.5:
                print(f"  NOTE: X and Y voxels are different - is that expected?")
            if 0.5 <= vx <= 20 and 0.5 <= vy <= 20 and 0.5 <= vz <= 20:
                print("  OK: Values look reasonable for lightsheet microscopy")
    else:
        print("\n  Could not extract voxel sizes from file!")
    
    print("=" * 60)
    
    # Save full dump
    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.write('\n'.join(lines))
    
    print(f"\nFull metadata saved to: {output_path}")
    return output_path, voxel_info


def main():
    print("=" * 60)
    print("IMS Metadata Dumper")
    print("=" * 60)
    
    # Find all .ims files
    root = DEFAULT_BRAINGLOBE_ROOT
    if not root.exists():
        print(f"\nDefault path not found: {root}")
        root = Path(input("Enter path to search: ").strip().strip('"'))
    
    print(f"\nScanning {root} for .ims files...")
    ims_files = find_ims_files(root)
    
    if not ims_files:
        print("No .ims files found!")
        return
    
    # Show list
    print(f"\nFound {len(ims_files)} .ims file(s):\n")
    for i, f in enumerate(ims_files, 1):
        # Show relative path for cleaner display
        try:
            rel = f.relative_to(root)
        except:
            rel = f
        size_gb = f.stat().st_size / (1024**3)
        print(f"  {i}. {rel} ({size_gb:.1f} GB)")
    
    # Pick one
    print()
    choice = input("Enter number to dump (or 'q' to quit): ").strip()
    
    if choice.lower() == 'q':
        return
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(ims_files):
            dump_ims_metadata(ims_files[idx])
        else:
            print("Invalid selection")
    except ValueError:
        print("Invalid input")


if __name__ == '__main__':
    main()
    print()
    input("Press Enter to close...")
