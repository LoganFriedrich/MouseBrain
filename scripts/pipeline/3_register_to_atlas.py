#!/usr/bin/env python3
r"""
3_register_to_atlas.py (v2.0.0)

================================================================================
WHAT IS THIS?
================================================================================
This is Script 3 in the BrainGlobe pipeline. It registers your cropped brain
images to the Allen Mouse Brain Atlas using brainreg.

Think of it as: "Align my brain to the atlas so we know where everything is"

Run this AFTER Script 2 (extract_and_analyze.py).
Run this BEFORE Script 4 (cell detection).

================================================================================
WHAT IT DOES
================================================================================
1. Reads metadata from Script 2 (channel roles, voxel sizes)
2. Runs brainreg to register your data to the Allen atlas
3. Archives any previous registration attempts (never overwrites)
4. Generates QC images for verification
5. Prepares output for cellfinder

================================================================================
HOW TO RUN
================================================================================
Open Anaconda Prompt, then:

    conda activate brainglobe-env
    cd <your-repo-path>/3_Nuclei_Detection/util_Scripts
    python 3_register_to_atlas.py

The script will show an interactive menu of available brains.

================================================================================
INTERACTIVE MENU
================================================================================
When you run the script, you'll see something like:

    [READY TO REGISTER]
      1. 349_CNT_01_02/349_CNT_01_02_1p625x_z4
      2. 350_SCI_02_05/350_SCI_02_05_1p9x_z3p37

    [ALREADY REGISTERED]
      3. 348_CNT_01_01/348_CNT_01_01_1p625x_z4

    Options:
      - Enter numbers to process (e.g., '1' or '1,2')
      - 'all' to process all ready
      - 'reprocess' to redo completed ones
      - 'q' to quit

================================================================================
REGISTRATION ARCHIVING
================================================================================
When you re-register a brain, the old registration is NOT deleted. Instead:

    3_Registered_Atlas/
    ├── registered_atlas.tiff      ← Current registration
    ├── brainreg.json
    ├── QC_registration.png
    └── _archive/
        ├── 20241216_143052/       ← Previous attempt #1
        └── 20241216_160215/       ← Previous attempt #2

You can always roll back by moving files from _archive back to main folder.

================================================================================
COMMAND LINE OPTIONS
================================================================================
    python 3_register_to_atlas.py              # Interactive mode
    python 3_register_to_atlas.py --batch      # Process all pending automatically
    python 3_register_to_atlas.py --inspect    # Dry run - show status only
    python 3_register_to_atlas.py --n-free-cpus 106  # Limit CPU usage

================================================================================
SETTINGS (edit in script if needed)
================================================================================
    DEFAULT_ATLAS = "allen_mouse_10um"   # Atlas to use
    DEFAULT_N_FREE_CPUS = 4              # CPUs to leave free

================================================================================
REQUIREMENTS
================================================================================
    pip install brainreg brainglobe-napari-io
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# =============================================================================
# VERSION
# =============================================================================
SCRIPT_VERSION = "2.0.2"

# =============================================================================
# PROGRESS HELPERS
# =============================================================================
def timestamp():
    """Get current time as formatted string."""
    return datetime.now().strftime("%H:%M:%S")

# =============================================================================
# VERIFY BRAINREG AVAILABILITY
# =============================================================================
def find_brainreg_executable():
    """Find brainreg executable, checking PATH and conda Scripts folder."""
    import shutil

    # First check PATH
    brainreg_path = shutil.which("brainreg")
    if brainreg_path:
        return brainreg_path

    # If not in PATH, check if we're running from a conda env
    # The brainreg.exe should be in the same env's Scripts folder
    python_exe = Path(sys.executable)
    if python_exe.parent.name.lower() in ('scripts', 'bin'):
        # We're running from Scripts folder - parent is the env
        env_scripts = python_exe.parent
    else:
        # Normal case - python.exe is in env root, Scripts is sibling
        env_scripts = python_exe.parent / "Scripts"

    brainreg_exe = env_scripts / "brainreg.exe"
    if brainreg_exe.exists():
        return str(brainreg_exe)

    # Also check Unix-style (for Linux/Mac)
    brainreg_unix = env_scripts / "brainreg"
    if brainreg_unix.exists():
        return str(brainreg_unix)

    # Check common conda locations
    for conda_base in [
        Path("G:/Program_Files/Conda"),
        Path("C:/Users") / os.environ.get("USERNAME", "") / "anaconda3",
        Path("C:/Users") / os.environ.get("USERNAME", "") / "miniconda3",
        Path(os.environ.get("CONDA_PREFIX", "")) if os.environ.get("CONDA_PREFIX") else None,
    ]:
        if conda_base and conda_base.exists():
            for env_name in ["brainglobe-env", "brainglobe", "cellfinder"]:
                scripts_path = conda_base / "envs" / env_name / "Scripts" / "brainreg.exe"
                if scripts_path.exists():
                    return str(scripts_path)

    return None


def check_brainreg_available():
    """Check if brainreg is available and return its path."""
    brainreg_path = find_brainreg_executable()

    if brainreg_path is None:
        print("=" * 70)
        print("ERROR: brainreg not found!")
        print("=" * 70)
        print()
        print("brainreg executable was not found. This usually means you need to")
        print("either activate the brainglobe conda environment, or run with")
        print("the brainglobe environment's Python directly.")
        print()
        print("Option 1 - Activate environment first:")
        print("    conda activate brainglobe-env")
        print("    python 3_register_to_atlas.py")
        print()
        print("Option 2 - Run directly with brainglobe Python:")
        print("    G:\\Program_Files\\Conda\\envs\\brainglobe-env\\python.exe 3_register_to_atlas.py")
        print()
        print(f"Current Python: {sys.executable}")
        print()
        sys.exit(1)

    return brainreg_path

# Check brainreg on import
_brainreg_path = check_brainreg_available()

# =============================================================================
# DEFAULT SETTINGS
# =============================================================================
from mousebrain.config import BRAINS_ROOT as DEFAULT_BRAINGLOBE_ROOT

# Import experiment tracker for logging registration runs
try:
    from mousebrain.tracker import ExperimentTracker
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False
    print("Warning: mousebrain.tracker not available - registration runs will not be logged")

# Atlas to use (allen_mouse_10um or allen_mouse_25um)
DEFAULT_ATLAS = "allen_mouse_10um"

# CPUs to leave free (for servers with many cores)
# IMPORTANT: Windows limits ProcessPoolExecutor to 61 workers max
# On a 104-core machine, we need at least 43 cores free (104-61=43)
# Using 50 to be safe and leave resources for other processes
DEFAULT_N_FREE_CPUS = 50  # Leave 50 cores free to stay under Windows 61-worker limit

# =============================================================================
# PIPELINE FOLDER NAMES (must match Scripts 1 & 2)
# =============================================================================
FOLDER_RAW_IMS = "0_Raw_IMS"
FOLDER_EXTRACTED_FULL = "1_Extracted_Full"
FOLDER_CROPPED = "2_Cropped_For_Registration"
FOLDER_REGISTRATION = "3_Registered_Atlas"

# Files that indicate a complete registration
REGISTRATION_REQUIRED_FILES = [
    "brainreg.json",
]


# =============================================================================
# DISCOVERY AND STATUS
# =============================================================================

def find_extracted_pipelines(root_path):
    """
    Find all pipelines that have been extracted (have cropped data).
    Returns list of (pipeline_folder, mouse_folder, metadata) tuples.
    """
    root_path = Path(root_path)
    pipelines = []
    
    for mouse_dir in root_path.iterdir():
        if not mouse_dir.is_dir() or mouse_dir.name.startswith('.'):
            continue
        if any(skip in mouse_dir.name.lower() for skip in ['script', 'backup', 'archive']):
            continue
        
        for pipeline_dir in mouse_dir.iterdir():
            if not pipeline_dir.is_dir():
                continue
            
            # Check for cropped data with metadata
            crop_folder = pipeline_dir / FOLDER_CROPPED
            metadata_path = crop_folder / "metadata.json"
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Verify cropped data exists
                    ch0 = crop_folder / "ch0"
                    if ch0.exists() and len(list(ch0.glob('Z*.tif'))) > 0:
                        pipelines.append((pipeline_dir, mouse_dir, metadata))
                except:
                    pass
    
    return pipelines


def check_registration_status(pipeline_folder):
    """
    Check if registration has been done.
    
    Returns:
        (status, reason)
        
        Status:
        - "needs_registration": Not registered yet
        - "incomplete": Started but failed/incomplete
        - "complete": Successfully registered
    """
    pipeline_folder = Path(pipeline_folder)
    reg_folder = pipeline_folder / FOLDER_REGISTRATION
    
    if not reg_folder.exists():
        return "needs_registration", "registration folder missing"
    
    # Get contents, ignoring hidden files and archive
    contents = [f for f in reg_folder.iterdir() 
                if not f.name.startswith('.') 
                and f.name.lower() not in ['thumbs.db', 'desktop.ini']
                and f.name != '_archive']
    
    if len(contents) == 0:
        return "needs_registration", "registration folder empty"
    
    # Check for required files
    missing = []
    for req_file in REGISTRATION_REQUIRED_FILES:
        if not (reg_folder / req_file).exists():
            missing.append(req_file)
    
    if missing:
        return "incomplete", f"missing: {', '.join(missing)}"
    
    return "complete", "registration complete"


# =============================================================================
# ARCHIVING
# =============================================================================

def archive_existing_registration(pipeline_folder):
    """
    Move existing registration to _archive folder with timestamp.
    Returns True if anything was archived.
    """
    reg_folder = Path(pipeline_folder) / FOLDER_REGISTRATION
    
    if not reg_folder.exists():
        return False
    
    # Get contents to archive (excluding _archive folder itself)
    contents = [f for f in reg_folder.iterdir() 
                if f.name != '_archive' 
                and not f.name.startswith('.')
                and f.name.lower() not in ['thumbs.db', 'desktop.ini']]
    
    if not contents:
        return False
    
    # Create archive folder
    archive_base = reg_folder / "_archive"
    archive_base.mkdir(exist_ok=True)
    
    # Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_folder = archive_base / timestamp
    archive_folder.mkdir()
    
    # Move contents
    for item in contents:
        dest = archive_folder / item.name
        shutil.move(str(item), str(dest))
    
    return True


# =============================================================================
# REGISTRATION
# =============================================================================

def run_brainreg(pipeline_folder, metadata, n_free_cpus=None, atlas=DEFAULT_ATLAS):
    """
    Run brainreg on a pipeline.

    Uses metadata from Script 2 to determine:
    - Which channel to register (background channel preferred)
    - Voxel sizes
    - Orientation

    Logs the registration run to the experiment tracker if available.
    """
    pipeline_folder = Path(pipeline_folder)

    crop_folder = pipeline_folder / FOLDER_CROPPED
    reg_folder = pipeline_folder / FOLDER_REGISTRATION

    # Ensure registration folder exists
    reg_folder.mkdir(parents=True, exist_ok=True)

    # Get registration channel from metadata
    # Prefer background channel for registration (clearer tissue boundaries)
    channels_info = metadata.get('channels', {})
    background_ch = channels_info.get('background_channel')
    signal_ch = channels_info.get('signal_channel', 0)

    reg_channel = background_ch if background_ch is not None else signal_ch
    input_folder = crop_folder / f"ch{reg_channel}"

    # Get voxel sizes
    voxel = metadata.get('voxel_size_um', {})
    voxel_z = voxel.get('z', 4.0)
    voxel_xy = voxel.get('x', 4.0)

    # Get orientation
    orientation = metadata.get('orientation', 'iar')

    # Get brain name from pipeline folder
    brain_name = pipeline_folder.name

    # Log registration start to tracker
    exp_id = None
    tracker = None
    if TRACKER_AVAILABLE:
        try:
            tracker = ExperimentTracker()
            exp_id = tracker.log_registration(
                brain=brain_name,
                atlas=atlas,
                orientation=orientation,
                voxel_z=voxel_z,
                voxel_xy=voxel_xy,
                input_path=str(input_folder),
                output_path=str(reg_folder),
                status="started",
                script_version=SCRIPT_VERSION,
            )
            print(f"    [Tracker] Registration run logged: {exp_id}")
        except Exception as e:
            print(f"    [Tracker] Warning: Could not log registration start: {e}")

    # Build brainreg command (use found path, not just "brainreg")
    cmd = [
        _brainreg_path,
        str(input_folder),
        str(reg_folder),
        "-v", str(voxel_z), str(voxel_xy), str(voxel_xy),
        "--orientation", orientation,
        "--atlas", atlas,
    ]

    # Add additional channels if present
    num_channels = channels_info.get('count', 1)
    if num_channels > 1:
        for ch_idx in range(num_channels):
            if ch_idx != reg_channel:
                additional_folder = crop_folder / f"ch{ch_idx}"
                if additional_folder.exists():
                    cmd.extend(["-a", str(additional_folder)])

    # Add CPU limit if specified
    if n_free_cpus is not None:
        cmd.extend(["--n-free-cpus", str(n_free_cpus)])

    print(f"    Registration channel: ch{reg_channel}")
    print(f"    Voxel sizes (Z,Y,X): {voxel_z}, {voxel_xy}, {voxel_xy} um")
    print(f"    Atlas: {atlas}")
    print(f"    Orientation: {orientation}")
    if n_free_cpus:
        print(f"    CPU limit: leaving {n_free_cpus} cores free")
    print(f"\n    [{timestamp()}] Starting brainreg (this typically takes 20-60 minutes)...")
    print(f"    Command: {' '.join(cmd)}")
    print(f"    {'='*50}")
    sys.stdout.flush()

    # Run brainreg
    start_time = datetime.now()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"    {'='*50}")
    if result.returncode != 0:
        # Update tracker with failure
        if tracker and exp_id:
            try:
                tracker.update_status(exp_id, status="failed", duration_seconds=elapsed)
            except Exception:
                pass
        raise RuntimeError(f"brainreg failed with return code {result.returncode}")

    print(f"    [{timestamp()}] Registration completed in {elapsed/60:.1f} minutes")

    # Verify output
    status, reason = check_registration_status(pipeline_folder)
    if status != "complete":
        # Update tracker with failure
        if tracker and exp_id:
            try:
                tracker.update_status(exp_id, status="failed", duration_seconds=elapsed)
            except Exception:
                pass
        raise RuntimeError(f"Registration verification failed: {reason}")

    # Update tracker with success
    if tracker and exp_id:
        try:
            tracker.update_status(exp_id, status="completed", duration_seconds=elapsed)
            print(f"    [Tracker] Registration logged as completed")
        except Exception as e:
            print(f"    [Tracker] Warning: Could not update status: {e}")

    # Generate detailed QC visualization
    print(f"    [{timestamp()}] Generating registration QC images...")
    qc_paths = []
    try:
        # Note: Path is already imported at module level, don't re-import here
        util_scripts = Path(__file__).parent
        if str(util_scripts) not in sys.path:
            sys.path.insert(0, str(util_scripts))
        from util_registration_qc import load_registration_data, create_qc_visualization

        qc_data = load_registration_data(pipeline_folder)
        if qc_data:
            qc_path = create_qc_visualization(qc_data, num_levels=5)
            qc_paths.append(str(qc_path))
            print(f"    [{timestamp()}] QC images saved: {qc_path.name}")

            # Add QC image paths to tracker
            if tracker and exp_id:
                try:
                    tracker.add_registration_qc_images(exp_id, qc_paths)
                except Exception:
                    pass
        else:
            print(f"    [{timestamp()}] Warning: Could not generate QC images")
    except Exception as e:
        print(f"    [{timestamp()}] Warning: QC generation failed: {e}")
        # Don't fail registration if QC generation fails

    return True


# =============================================================================
# INTERACTIVE MENU
# =============================================================================

def interactive_select_pipelines(pipelines, statuses):
    """
    Show interactive menu for selecting which pipelines to process.
    
    Returns list of indices to process, or None to cancel.
    """
    # Categorize pipelines
    ready = []
    complete = []
    incomplete = []
    
    for i, ((pipeline_folder, mouse_folder, metadata), (status, reason)) in enumerate(zip(pipelines, statuses)):
        display_name = f"{mouse_folder.name}/{pipeline_folder.name}"
        if status == "needs_registration":
            ready.append((i, display_name))
        elif status == "complete":
            complete.append((i, display_name))
        else:
            incomplete.append((i, display_name, reason))
    
    # Display menu
    print("\n" + "=" * 60)
    print("PIPELINE STATUS")
    print("=" * 60)
    
    if ready:
        print("\n[READY TO REGISTER]")
        for idx, name in ready:
            print(f"  {idx + 1}. {name}")
    
    if complete:
        print("\n[ALREADY REGISTERED]")
        for idx, name in complete:
            print(f"  {idx + 1}. {name}")
    
    if incomplete:
        print("\n[INCOMPLETE/FAILED]")
        for idx, name, reason in incomplete:
            print(f"  {idx + 1}. {name} ({reason})")
    
    print("\n" + "-" * 60)
    print("Options:")
    print("  - Enter numbers to process (e.g., '1' or '1,2,3')")
    print("  - 'all' to process all ready")
    print("  - 'reprocess' to redo already-completed")
    print("  - 'retry' to retry incomplete/failed")
    print("  - 'q' to quit")
    print("-" * 60)
    
    while True:
        response = input("\nSelection: ").strip().lower()
        
        if response == 'q':
            return None
        
        if response == 'all':
            if not ready:
                print("No pipelines ready to register.")
                continue
            return [idx for idx, _ in ready]
        
        if response == 'reprocess':
            if not complete:
                print("No completed registrations to reprocess.")
                continue
            return [idx for idx, _ in complete]
        
        if response == 'retry':
            if not incomplete:
                print("No incomplete registrations to retry.")
                continue
            return [idx for idx, _, _ in incomplete]
        
        # Parse comma-separated numbers
        try:
            indices = []
            for part in response.split(','):
                num = int(part.strip()) - 1  # Convert to 0-indexed
                if 0 <= num < len(pipelines):
                    indices.append(num)
                else:
                    print(f"Invalid number: {num + 1}")
                    indices = None
                    break
            
            if indices:
                return indices
        except ValueError:
            print("Invalid input. Enter numbers, 'all', 'reprocess', 'retry', or 'q'.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Register brain images to atlas using brainreg',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Version: {SCRIPT_VERSION}

This script registers your cropped brain data to the Allen atlas.
Run AFTER Script 2 (extract_and_analyze.py).

Examples:
  python 3_register_to_atlas.py
  python 3_register_to_atlas.py --batch
  python 3_register_to_atlas.py --n-free-cpus 106
        """
    )
    
    parser.add_argument('path', nargs='?', default=None,
                        help=f'Path to scan (default: {DEFAULT_BRAINGLOBE_ROOT})')
    parser.add_argument('--inspect', '-i', action='store_true',
                        help='Dry run - show status only')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Non-interactive - process all pending')
    parser.add_argument('--n-free-cpus', type=int, default=DEFAULT_N_FREE_CPUS,
                        help=f'CPUs to leave free (default: {DEFAULT_N_FREE_CPUS})')
    parser.add_argument('--atlas', default=DEFAULT_ATLAS,
                        help=f'Atlas to use (default: {DEFAULT_ATLAS})')
    parser.add_argument('--brain', type=str, default=None,
                        help='Process only this specific brain (partial name match)')

    args = parser.parse_args()
    
    root_path = Path(args.path) if args.path else DEFAULT_BRAINGLOBE_ROOT
    
    if not root_path.exists():
        print(f"Error: Path not found: {root_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("BrainGlobe Atlas Registration")
    print(f"Version: {SCRIPT_VERSION}")
    print(f"Atlas: {args.atlas}")
    if args.n_free_cpus:
        print(f"CPU Limit: leaving {args.n_free_cpus} cores free")
    print("=" * 60)
    
    # Find extracted pipelines
    print(f"\nScanning: {root_path}")
    pipelines = find_extracted_pipelines(root_path)
    
    if not pipelines:
        print("\nNo extracted pipelines found.")
        print("Run Script 2 (extract_and_analyze.py) first!")
        return
    
    # Check status of each
    statuses = []
    for pipeline_folder, mouse_folder, metadata in pipelines:
        status, reason = check_registration_status(pipeline_folder)
        statuses.append((status, reason))
    
    if args.inspect:
        # Just show status
        print(f"\nFound {len(pipelines)} extracted pipeline(s):\n")
        for (pipeline_folder, mouse_folder, metadata), (status, reason) in zip(pipelines, statuses):
            symbol = "[OK]" if status == "complete" else "[  ]" if status == "needs_registration" else "[!!]"
            print(f"  {symbol} {mouse_folder.name}/{pipeline_folder.name} - {reason}")
        return
    
    # Select pipelines to process
    if args.batch:
        # Non-interactive: process all pending
        to_process = [i for i, (status, _) in enumerate(statuses)
                      if status in ["needs_registration", "incomplete"]]

        # Filter by brain name if specified
        if args.brain:
            to_process = [i for i in to_process
                         if args.brain in pipelines[i][0].name]
            if not to_process:
                # Also check if already complete and just needs to be reported
                for i, (pipeline_folder, mouse_folder, metadata) in enumerate(pipelines):
                    if args.brain in pipeline_folder.name:
                        status, _ = statuses[i]
                        if status == "complete":
                            print(f"\nBrain '{args.brain}' is already registered.")
                            return
                print(f"\nNo pending registration found for brain matching '{args.brain}'")
                return

        if not to_process:
            print("\nNo pipelines need registration. All complete!")
            return
    else:
        # Interactive selection
        to_process = interactive_select_pipelines(pipelines, statuses)
        if to_process is None:
            print("Cancelled.")
            return
        if not to_process:
            print("Nothing selected.")
            return
    
    # Confirm
    print(f"\nWill process {len(to_process)} pipeline(s):")
    for idx in to_process:
        pipeline_folder, mouse_folder, metadata = pipelines[idx]
        print(f"  - {mouse_folder.name}/{pipeline_folder.name}")
    
    if not args.batch:
        response = input("\nProceed? [Enter to continue, 'q' to quit]: ").strip()
        if response.lower() == 'q':
            print("Cancelled.")
            return
    
    # Process
    print("\n" + "=" * 60)
    print("Processing...")
    print("=" * 60)
    
    success = 0
    failed = 0
    
    for i, idx in enumerate(to_process):
        pipeline_folder, mouse_folder, metadata = pipelines[idx]
        status, _ = statuses[idx]
        
        print(f"\n[{i+1}/{len(to_process)}] {mouse_folder.name}/{pipeline_folder.name}")
        print("-" * 50)
        
        try:
            # Archive existing if reprocessing
            if status == "complete":
                print("    Archiving previous registration...")
                archived = archive_existing_registration(pipeline_folder)
                if archived:
                    print("    [OK] Previous registration archived")
            
            # Run registration
            run_brainreg(
                pipeline_folder, 
                metadata,
                n_free_cpus=args.n_free_cpus,
                atlas=args.atlas
            )
            
            success += 1
            print(f"\n    [OK] Registration complete")
            
        except Exception as e:
            failed += 1
            print(f"\n    [FAIL] Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Complete: {success} succeeded, {failed} failed")
    print("=" * 60)

    if success > 0:
        print("\n" + "=" * 60)
        print("WHAT TO DO NEXT")
        print("=" * 60)
        print("\n1. REVIEW the registration QC images:")
        print("   Look in each brain's 3_Registered_Atlas folder for:")
        print("   - QC_registration_detailed.png")
        print("\n2. APPROVE good registrations:")
        print("   python util_approve_registration.py --brain BRAIN_NAME")
        print("\n3. Then run cell detection:")
        print("   python 4_detect_cells.py --brain BRAIN_NAME")
        print("\n" + "-" * 60)
        print("OR just run: python RUN_PIPELINE.py")
        print("   (it will guide you through everything)")
        print("=" * 60)

    # Return non-zero exit code if any failures
    return 1 if failed > 0 else 0


if __name__ == '__main__':
    exit_code = main()

    if len(sys.argv) == 1:
        print()
        input("Press Enter to close...")

    sys.exit(exit_code)
