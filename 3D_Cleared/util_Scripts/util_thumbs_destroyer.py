#!/usr/bin/env python3
r"""
util_thumbs_destroyer.py

Nuclear option for thumbs.db files.

These files are created by Windows when viewing folders with images.
They prevent drag-and-drop into napari and are notoriously difficult to delete
because Windows marks them as hidden+system and keeps them locked.

This script:
1. Catalogs ALL open Explorer windows
2. Kills explorer.exe (releases all file locks)
3. Finds and destroys all thumbs.db files recursively
4. Restarts explorer.exe
5. Reopens all your Explorer windows exactly as they were
6. Optionally creates read-only decoy files to prevent recreation

Double-click usage:
    - Place this script in a PARENT folder of the images you want to clean
    - Double-click - it handles everything automatically
    - Your Explorer windows will briefly close and reopen

Command-line usage:
    python util_thumbs_destroyer.py                    # Script's folder (recursive)
    python util_thumbs_destroyer.py Y:\path\to\folder  # Specific path
    python util_thumbs_destroyer.py --prevent          # Also create decoys
    python util_thumbs_destroyer.py --dry-run          # Preview only (no Explorer kill)

Requirements:
    None (pure Python stdlib)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# =============================================================================
# EXPLORER WINDOW MANAGEMENT (Windows only)
# =============================================================================

def get_all_explorer_windows():
    """
    Get ALL currently open Explorer windows and their paths.
    Returns list of path strings.
    """
    if sys.platform != 'win32':
        return []
    
    try:
        # PowerShell command to get all Explorer window locations
        ps_cmd = '''
        $shell = New-Object -ComObject Shell.Application
        foreach ($w in $shell.Windows()) {
            try {
                $loc = $w.LocationURL
                if ($loc) {
                    Write-Output $loc
                }
            } catch {}
        }
        '''
        result = subprocess.run(
            ['powershell', '-Command', ps_cmd],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        import urllib.parse
        paths = []
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Convert file URL to path
            # Local: file:///C:/path -> C:\path
            # UNC: file://///server/share/path -> \\server\share\path
            # Or: file://server/share/path -> \\server\share\path
            if line.startswith('file://///'):
                path = '\\\\' + line[10:].replace('/', '\\')
            elif line.startswith('file:////'):
                path = '\\\\' + line[9:].replace('/', '\\')
            elif line.startswith('file:///'):
                path = line[8:].replace('/', '\\')
            elif line.startswith('file://'):
                path = '\\\\' + line[7:].replace('/', '\\')
            else:
                continue
            
            # URL decode (%20 for space, etc.)
            path = urllib.parse.unquote(path)
            paths.append(path)
        
        return paths
    except Exception as e:
        return []


def kill_explorer():
    """Kill explorer.exe process."""
    if sys.platform != 'win32':
        return False
    
    try:
        result = subprocess.run(
            ['taskkill', '/F', '/IM', 'explorer.exe'],
            capture_output=True,
            timeout=10
        )
        import time
        time.sleep(0.5)  # Give it a moment to die
        return True
    except Exception:
        return False


def start_explorer():
    """Restart explorer.exe process."""
    if sys.platform != 'win32':
        return False
    
    try:
        subprocess.Popen(['explorer.exe'], shell=False)
        import time
        time.sleep(1)  # Give it a moment to start
        return True
    except Exception:
        return False


def open_explorer_window(path):
    """Open an Explorer window at the given path."""
    if sys.platform != 'win32':
        return
    
    try:
        subprocess.Popen(['explorer', str(path)], shell=False)
        import time
        time.sleep(0.2)  # Small delay between opening windows
    except Exception:
        pass


def strip_windows_attributes(filepath):
    """Remove hidden and system attributes on Windows."""
    if sys.platform == 'win32':
        try:
            subprocess.run(
                ['attrib', '-h', '-s', '-r', str(filepath)],
                capture_output=True,
                check=False,
                timeout=5
            )
            return True
        except Exception:
            return False
    return True


def set_readonly_hidden(filepath):
    """Set file as read-only and hidden on Windows."""
    if sys.platform == 'win32':
        try:
            subprocess.run(
                ['attrib', '+h', '+r', str(filepath)],
                capture_output=True,
                check=False,
                timeout=5
            )
            return True
        except Exception:
            return False
    return True


def find_thumbs_files(root_path):
    """Find all thumbs.db files (case-insensitive)."""
    root_path = Path(root_path)
    thumbs_files = []
    
    for path in root_path.rglob('*'):
        if path.is_file() and path.name.lower() == 'thumbs.db':
            thumbs_files.append(path)
    
    return thumbs_files


def destroy_thumbs(filepath, dry_run=False):
    """
    Destroy a single thumbs.db file.
    Returns (success, message).
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return True, "already gone"
    
    if dry_run:
        return True, "would delete"
    
    try:
        # Strip attributes first (Windows)
        strip_windows_attributes(filepath)
        
        # Try normal delete
        filepath.unlink()
        return True, "destroyed"
        
    except PermissionError:
        # Try harder on Windows
        if sys.platform == 'win32':
            try:
                # Force delete via cmd
                result = subprocess.run(
                    ['cmd', '/c', 'del', '/f', '/q', str(filepath)],
                    capture_output=True,
                    check=False,
                    timeout=5
                )
                if not filepath.exists():
                    return True, "force destroyed"
            except Exception:
                pass
        return False, "permission denied (file may be locked by Explorer)"
        
    except Exception as e:
        return False, str(e)


def create_decoy(folder_path):
    """
    Create a read-only decoy thumbs.db file.
    Windows won't overwrite a read-only file.
    """
    folder_path = Path(folder_path)
    decoy_path = folder_path / "Thumbs.db"
    
    if decoy_path.exists():
        return False, "file already exists"
    
    try:
        # Create empty file
        decoy_path.write_bytes(b'')
        
        # Make it read-only and hidden
        set_readonly_hidden(decoy_path)
        
        return True, "decoy created"
    except Exception as e:
        return False, str(e)


def find_image_folders(root_path):
    """Find folders that contain image files (likely targets for thumbs.db)."""
    image_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    root_path = Path(root_path)
    image_folders = set()
    
    for path in root_path.rglob('*'):
        if path.is_file() and path.suffix.lower() in image_extensions:
            image_folders.add(path.parent)
    
    return sorted(image_folders)


def main():
    parser = argparse.ArgumentParser(
        description='Destroy all thumbs.db files. Nuke them from orbit.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python util_thumbs_destroyer.py
    python util_thumbs_destroyer.py Y:\\2_Connectome\\Brainglobe
    python util_thumbs_destroyer.py --prevent
    python util_thumbs_destroyer.py --dry-run
    
Why this exists:
    thumbs.db files are created by Windows Explorer when you view a folder
    containing images. They cache thumbnail data. Unfortunately:
    
    1. They prevent napari from accepting drag-and-drop of folders
    2. Windows marks them as hidden+system, making them hard to delete
    3. Windows locks them while Explorer has viewed the folder
    4. They keep coming back
    
How it works:
    This script takes the nuclear option: it saves all your open Explorer
    windows, kills explorer.exe entirely to release ALL file locks, deletes
    the thumbs.db files, restarts Explorer, and reopens all your windows.
    
    Optionally creates read-only decoys to prevent Windows from recreating them.
    
Double-click mode:
    When double-clicked (no arguments), destroys thumbs.db in the folder
    where this script is located and all subfolders.
        """
    )
    
    parser.add_argument('path', nargs='?', default=None,
                        help='Path to scan (default: folder where this script is located)')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Show what would be deleted without actually deleting')
    parser.add_argument('--prevent', '-p', action='store_true',
                        help='Create read-only decoy files to prevent thumbs.db recreation')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')
    
    args = parser.parse_args()
    
    # If no path specified, use the script's directory (for double-click mode)
    if args.path is None:
        root_path = Path(__file__).parent.resolve()
    else:
        root_path = Path(args.path).resolve()
    
    if not root_path.exists():
        print(f"Error: Path not found: {root_path}")
        sys.exit(1)
    
    if not args.quiet:
        print("=" * 60)
        print("THUMBS.DB DESTROYER")
        print("=" * 60)
        print(f"\nTarget: {root_path}")
        if args.dry_run:
            print("Mode: DRY RUN (no files will be deleted)")
        print()
    
    # Save all Explorer windows, kill explorer, do the work, restart and reopen
    saved_windows = []
    explorer_killed = False
    
    if not args.dry_run and sys.platform == 'win32':
        if not args.quiet:
            print("Cataloging open Explorer windows...")
        saved_windows = get_all_explorer_windows()
        if saved_windows:
            if not args.quiet:
                print(f"  Found {len(saved_windows)} window(s):")
                for w in saved_windows:
                    print(f"    - {w}")
                print()
                print("Killing Explorer to release file locks...")
            
            if kill_explorer():
                explorer_killed = True
                if not args.quiet:
                    print("  Explorer killed.")
                    print()
            else:
                if not args.quiet:
                    print("  WARNING: Failed to kill Explorer. Some files may still be locked.")
                    print()
        else:
            if not args.quiet:
                print("  No Explorer windows open.")
                print()
    
    # Find all thumbs.db files
    if not args.quiet:
        print("Scanning for thumbs.db files...")
    
    thumbs_files = find_thumbs_files(root_path)
    
    if not thumbs_files:
        if not args.quiet:
            print("  No thumbs.db files found. The area is clean.")
    else:
        if not args.quiet:
            print(f"  Found {len(thumbs_files)} thumbs.db file(s)")
            print()
            print("Destroying...")
        
        destroyed = 0
        failed = 0
        
        for thumbs in thumbs_files:
            rel_path = thumbs.relative_to(root_path) if thumbs.is_relative_to(root_path) else thumbs
            success, msg = destroy_thumbs(thumbs, dry_run=args.dry_run)
            
            if success:
                destroyed += 1
                if not args.quiet:
                    print(f"  ✓ {rel_path} ({msg})")
            else:
                failed += 1
                if not args.quiet:
                    print(f"  ✗ {rel_path} ({msg})")
        
        if not args.quiet:
            print()
            if args.dry_run:
                print(f"Would destroy: {destroyed} file(s)")
            else:
                print(f"Destroyed: {destroyed} file(s)")
            if failed > 0:
                print(f"Failed: {failed} file(s) - try closing Explorer windows")
    
    # Create decoys if requested
    if args.prevent and not args.dry_run:
        if not args.quiet:
            print()
            print("Creating decoy files to prevent recreation...")
        
        image_folders = find_image_folders(root_path)
        decoys_created = 0
        
        for folder in image_folders:
            success, msg = create_decoy(folder)
            if success:
                decoys_created += 1
                if not args.quiet:
                    rel_path = folder.relative_to(root_path) if folder.is_relative_to(root_path) else folder
                    print(f"  ✓ {rel_path}/Thumbs.db (decoy)")
        
        if not args.quiet:
            print()
            print(f"Decoys created: {decoys_created}")
    
    # Restart Explorer and reopen windows if we killed it
    if explorer_killed:
        if not args.quiet:
            print()
            print("Restarting Explorer...")
        
        if start_explorer():
            if not args.quiet:
                print("  Explorer restarted.")
                
            if saved_windows:
                if not args.quiet:
                    print()
                    print("Reopening your windows...")
                for window_path in saved_windows:
                    open_explorer_window(window_path)
                    if not args.quiet:
                        print(f"  ↺ {window_path}")
        else:
            if not args.quiet:
                print("  WARNING: Failed to restart Explorer. Please start it manually.")
                print("  (Press Win+E or run 'explorer.exe')")
    
    if not args.quiet:
        print()
        print("=" * 60)
        print("Complete.")
        print("=" * 60)
        
        # Pause so window doesn't close immediately when double-clicked
        if args.path is None:  # Double-click mode
            print()
            input("Press Enter to close...")


if __name__ == '__main__':
    main()
