#!/usr/bin/env python3
"""
util_directory_dump.py

Dump directory structure for sharing/debugging.

Usage:
    python util_directory_dump.py                     # Dump parent of script location
    python util_directory_dump.py Y:\\path\\to\\dir   # Dump specific directory
    python util_directory_dump.py . --depth 4         # Current dir, deeper scan

Output saved to: DIRECTORY_DUMP.txt (in the target directory)

NO EXTERNAL DEPENDENCIES - uses only Python standard library.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def format_size(size_bytes):
    """Format file size in human-readable units."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/1024**2:.1f} MB"
    else:
        return f"{size_bytes/1024**3:.2f} GB"


def format_date(timestamp):
    """Format timestamp as readable date."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def get_file_info(path):
    """Get file metadata."""
    try:
        stat = path.stat()
        return {
            'size': stat.st_size,
            'size_human': format_size(stat.st_size),
            'modified': format_date(stat.st_mtime),
        }
    except:
        return {'size': 0, 'size_human': '?', 'modified': '?'}


def summarize_json(path):
    """Read and return JSON contents."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"(Could not read: {e})"


def print_file(f, prefix, output, is_last=False):
    """Print a single file with metadata."""
    connector = "└── " if is_last else "├── "
    info = get_file_info(f)
    
    # Icons by extension
    icons = {'.ims': '[IMS]', '.tif': '[TIF]', '.tiff': '[TIF]', '.json': '[JSON]', 
             '.txt': '[TXT]', '.py': '[PY]', '.ijm': '[IJM]', '.md': '[MD]', '.bat': '[BAT]'}
    icon = icons.get(f.suffix.lower(), '[FILE]')
    
    output.append(f"{prefix}{connector}{icon} {f.name}")
    output.append(f"{prefix}{'    ' if is_last else '│   '}   {info['size_human']} | {info['modified']}")
    
    # Show JSON contents
    if f.suffix.lower() == '.json':
        extension = "    " if is_last else "│   "
        json_content = summarize_json(f)
        for line in json_content.split('\n'):
            output.append(f"{prefix}{extension}   {line}")


def list_dir_contents(path, prefix, output, max_depth=2, current_depth=0, verbose_path=""):
    """List contents of a directory."""
    if current_depth > max_depth:
        output.append(f"{prefix}... (deeper levels not shown)")
        return
    
    try:
        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    except PermissionError:
        output.append(f"{prefix}(permission denied)")
        print(f"      [!] Permission denied: {verbose_path or path.name}")
        return
    except Exception as e:
        output.append(f"{prefix}(error: {e})")
        print(f"      [!] Error reading: {verbose_path or path.name}: {e}")
        return
    
    # Filter hidden files
    dirs = [i for i in items if i.is_dir() and not i.name.startswith('.')]
    files = [i for i in items if i.is_file() and not i.name.startswith('.')]
    
    all_items = dirs + files
    
    # Show progress for directories with many items
    if len(all_items) > 50 and current_depth == 0:
        print(f"      ({len(all_items)} items to process...)")
    
    for i, item in enumerate(all_items):
        is_last = (i == len(all_items) - 1)
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "
        
        if item.is_dir():
            # Count items
            try:
                subcount = len(list(item.iterdir()))
                subinfo = f" ({subcount} items)"
            except:
                subinfo = ""
            
            output.append(f"{prefix}{connector}[DIR] {item.name}/{subinfo}")
            list_dir_contents(item, prefix + extension, output, max_depth, current_depth + 1, f"{verbose_path}/{item.name}")
        else:
            print_file(item, prefix, output, is_last)


def main():
    parser = argparse.ArgumentParser(
        description='Dump directory structure for sharing/debugging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python util_directory_dump.py
    python util_directory_dump.py Y:\\2_Connectome\\Brainglobe
    python util_directory_dump.py . --depth 4
    python util_directory_dump.py C:\\Data --no-wait
        """
    )
    parser.add_argument('path', nargs='?', default=None,
                        help='Directory to dump (default: parent of script location)')
    parser.add_argument('--depth', '-d', type=int, default=2,
                        help='Max depth to scan into subdirectories (default: 2)')
    parser.add_argument('--no-wait', '-n', action='store_true',
                        help='Do not wait for Enter at the end')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path (default: DIRECTORY_DUMP.txt in target dir)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DIRECTORY DUMP UTILITY")
    print("=" * 60)
    print()
    
    # Where is this script?
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    
    # Determine target directory
    if args.path:
        target_dir = Path(args.path).resolve()
    else:
        target_dir = script_dir.parent  # Default: parent of script location
    
    if not target_dir.exists():
        print(f"ERROR: Path does not exist: {target_dir}")
        if not args.no_wait:
            input("\nPress Enter to exit...")
        return 1
    
    if not target_dir.is_dir():
        print(f"ERROR: Path is not a directory: {target_dir}")
        if not args.no_wait:
            input("\nPress Enter to exit...")
        return 1
    
    # Determine output file location
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = target_dir / "DIRECTORY_DUMP.txt"
    
    print(f"Script location: {script_path}")
    print(f"Target directory: {target_dir}")
    print(f"Output file: {output_file}")
    print(f"Scan depth: {args.depth}")
    print()
    
    # Open file for writing - we'll append as we go
    print(f"Opening output file for writing...")
    sys.stdout.flush()
    
    try:
        f = open(output_file, 'w', encoding='utf-8')
    except Exception as e:
        print(f"ERROR: Cannot open output file: {e}")
        alt_output = Path.home() / "DIRECTORY_DUMP.txt"
        print(f"Trying alternate location: {alt_output}")
        try:
            f = open(alt_output, 'w', encoding='utf-8')
            output_file = alt_output
        except Exception as e2:
            print(f"ERROR: Cannot write anywhere: {e2}")
            if not args.no_wait:
                input("\nPress Enter to exit...")
            return 1
    
    print(f"  Opened: {output_file}")
    lines_written = 0
    
    def write_line(line=""):
        nonlocal lines_written
        f.write(line + "\n")
        lines_written += 1
    
    def flush_file():
        f.flush()
    
    # =========================================================================
    # HEADER
    # =========================================================================
    write_line("=" * 80)
    write_line("DIRECTORY DUMP")
    write_line("=" * 80)
    write_line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write_line(f"Python: {sys.executable}")
    write_line(f"Target: {target_dir}")
    write_line(f"Scan depth: {args.depth}")
    write_line("")
    flush_file()
    
    # =========================================================================
    # SECTION 1: Path hierarchy (what's above target)
    # =========================================================================
    print("Section 1: Recording path hierarchy...")
    sys.stdout.flush()
    
    write_line("=" * 80)
    write_line("PATH HIERARCHY (parents of target)")
    write_line("=" * 80)
    
    parents = list(target_dir.parents)
    for i, p in enumerate(reversed(parents)):
        indent = "  " * i
        write_line(f"{indent}[DIR] {p.name or p}/")
    
    # Target directory
    indent = "  " * len(parents)
    write_line(f"{indent}[DIR] {target_dir.name}/  <-- TARGET DIRECTORY")
    write_line("")
    flush_file()
    print("  Done.")
    
    # =========================================================================
    # SECTION 2: Target directory contents (top level)
    # =========================================================================
    print(f"Section 2: Listing target directory contents ({target_dir.name})...")
    sys.stdout.flush()
    
    write_line("=" * 80)
    write_line(f"TARGET DIRECTORY CONTENTS: {target_dir}")
    write_line("=" * 80)
    
    try:
        print("  Reading directory listing...")
        sys.stdout.flush()
        items = sorted(target_dir.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        items = [s for s in items if not s.name.startswith('.')]
        
        print(f"  Found {len(items)} items")
        sys.stdout.flush()
        
        for i, item in enumerate(items):
            is_last = (i == len(items) - 1)
            connector = "└── " if is_last else "├── "
            
            if item.is_dir():
                try:
                    count = len(list(item.iterdir()))
                    write_line(f"{connector}[DIR] {item.name}/ ({count} items)")
                except:
                    write_line(f"{connector}[DIR] {item.name}/")
            else:
                info = get_file_info(item)
                write_line(f"{connector}[FILE] {item.name} ({info['size_human']})")
        
        flush_file()
        print("  Done.")
    except Exception as e:
        write_line(f"(error reading directory: {e})")
        print(f"  ERROR: {e}")
    
    write_line("")
    
    # =========================================================================
    # SECTION 3: Contents of each subdirectory (the main event)
    # =========================================================================
    print("Section 3: Dumping contents of each subdirectory...")
    sys.stdout.flush()
    
    write_line("=" * 80)
    write_line("CONTENTS OF EACH SUBDIRECTORY")
    write_line("=" * 80)
    flush_file()
    
    def list_dir_to_file(path, prefix, max_depth, current_depth=0):
        """List contents of a directory, writing directly to file."""
        if current_depth > max_depth:
            write_line(f"{prefix}... (deeper levels not shown)")
            return
        
        try:
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            write_line(f"{prefix}(permission denied)")
            return
        except Exception as e:
            write_line(f"{prefix}(error: {e})")
            return
        
        # Filter hidden files
        dirs = [i for i in items if i.is_dir() and not i.name.startswith('.')]
        files = [i for i in items if i.is_file() and not i.name.startswith('.')]
        
        all_items = dirs + files
        
        for i, item in enumerate(all_items):
            is_last = (i == len(all_items) - 1)
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "
            
            if item.is_dir():
                try:
                    subcount = len(list(item.iterdir()))
                    subinfo = f" ({subcount} items)"
                except:
                    subinfo = ""
                
                write_line(f"{prefix}{connector}[DIR] {item.name}/{subinfo}")
                list_dir_to_file(item, prefix + extension, max_depth, current_depth + 1)
            else:
                # Print file with metadata
                info = get_file_info(item)
                icons = {'.ims': '[IMS]', '.tif': '[TIF]', '.tiff': '[TIF]', '.json': '[JSON]', 
                         '.txt': '[TXT]', '.py': '[PY]', '.ijm': '[IJM]', '.md': '[MD]', '.bat': '[BAT]'}
                icon = icons.get(item.suffix.lower(), '[FILE]')
                
                write_line(f"{prefix}{connector}{icon} {item.name}")
                write_line(f"{prefix}{'    ' if is_last else '│   '}   {info['size_human']} | {info['modified']}")
                
                # Show JSON contents
                if item.suffix.lower() == '.json':
                    ext = "    " if is_last else "│   "
                    json_content = summarize_json(item)
                    for line in json_content.split('\n'):
                        write_line(f"{prefix}{ext}   {line}")
    
    try:
        items = sorted(target_dir.iterdir(), key=lambda x: x.name.lower())
        subdirs = [s for s in items if s.is_dir() and not s.name.startswith('.')]
        
        for idx, subdir in enumerate(subdirs):
            print(f"  [{idx+1}/{len(subdirs)}] Scanning: {subdir.name}/ ...")
            sys.stdout.flush()
            
            write_line("")
            write_line("-" * 80)
            write_line(f"[DIR] {subdir.name}/")
            write_line(f"   Full path: {subdir}")
            write_line("-" * 80)
            
            list_dir_to_file(subdir, "", max_depth=args.depth, current_depth=0)
            flush_file()  # Flush after each subdirectory
            
            print(f"       Done. ({lines_written} lines written so far)")
            sys.stdout.flush()
        
        print(f"  Finished all {len(subdirs)} subdirectories.")
    except Exception as e:
        write_line(f"(error: {e})")
        print(f"  ERROR: {e}")
    
    # =========================================================================
    # FOOTER
    # =========================================================================
    write_line("")
    write_line("=" * 80)
    write_line("END OF DUMP")
    write_line("=" * 80)
    
    f.close()
    
    print()
    print("=" * 60)
    print(f"DONE - Wrote {lines_written} lines to:")
    print(f"  {output_file}")
    print("=" * 60)
    
    if not args.no_wait:
        print()
        print(">>> Press Enter to close this window <<<")
        sys.stdout.flush()
        input()


if __name__ == '__main__':
    try:
        exit(main() or 0)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        exit(1)
