#!/usr/bin/env python3
"""
SCI-Connectome-Tools Installer

Automated installation script that sets up the complete pipeline environment.

Usage:
    python install.py

This will:
    1. Create/verify conda environment 'brainglobe-env'
    2. Install all required packages
    3. Install napari plugin for manual cropping
    4. Set up configuration
    5. Verify installation
"""

import os
import sys
import subprocess
import json
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

ENV_NAME = "brainglobe-env"
PYTHON_VERSION = "3.10"

# Core dependencies
CONDA_PACKAGES = [
    "python=3.10",
]

PIP_PACKAGES = [
    # BrainGlobe ecosystem
    "brainglobe-atlasapi",
    "brainreg>=1.0.0",
    "cellfinder>=1.0.0",
    "brainglobe-segmentation",

    # Imaging
    "napari[all]",
    "tifffile",
    "imageio",
    "scikit-image",

    # Data processing
    "numpy",
    "scipy",
    "pandas",
    "h5py",

    # Visualization
    "matplotlib",

    # IMS file support
    "imaris-ims-file-reader",

    # Utilities
    "tqdm",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_step(step_num, total_steps, text):
    """Print a step indicator."""
    print(f"\n[{step_num}/{total_steps}] {text}")
    print("-" * 70)


def run_command(cmd, check=True, capture_output=False):
    """Run a shell command and handle errors."""
    try:
        if capture_output:
            result = subprocess.run(
                cmd,
                shell=True,
                check=check,
                capture_output=True,
                text=True
            )
            return result
        else:
            subprocess.run(cmd, shell=True, check=check)
            return None
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed: {cmd}")
        if capture_output and e.stderr:
            print(f"Error: {e.stderr}")
        return None


def check_conda():
    """Check if conda is available."""
    result = run_command("conda --version", check=False, capture_output=True)
    if result and result.returncode == 0:
        version = result.stdout.strip()
        print(f"✓ Found conda: {version}")
        return True
    else:
        print("❌ Conda not found!")
        print("\nPlease install Miniconda from:")
        print("  https://docs.conda.io/en/latest/miniconda.html")
        return False


def check_env_exists():
    """Check if the environment already exists."""
    result = run_command(
        "conda env list",
        capture_output=True,
        check=False
    )
    if result and ENV_NAME in result.stdout:
        return True
    return False


def create_or_update_env():
    """Create or update the conda environment."""
    env_exists = check_env_exists()

    if env_exists:
        print(f"✓ Environment '{ENV_NAME}' already exists")
        response = input(f"\nUpdate existing environment? [Y/n]: ").strip().lower()
        if response and response != 'y':
            print("Skipping environment creation")
            return True
    else:
        print(f"Creating new environment '{ENV_NAME}'...")

    # Create environment with base packages
    if not env_exists:
        cmd = f"conda create -n {ENV_NAME} -y {' '.join(CONDA_PACKAGES)}"
        print(f"Running: {cmd}")
        result = run_command(cmd, check=False)
        if result is None:
            print("❌ Failed to create environment")
            return False
        print(f"✓ Environment created")

    return True


def install_packages():
    """Install all required packages via pip."""
    print("\nInstalling Python packages...")
    print("This will take several minutes. Please be patient...\n")

    # Determine the pip command for the environment
    if os.name == 'nt':  # Windows
        pip_cmd = f"conda run -n {ENV_NAME} pip"
    else:
        pip_cmd = f"conda run -n {ENV_NAME} pip"

    # Install packages
    packages_str = ' '.join(PIP_PACKAGES)
    cmd = f'{pip_cmd} install {packages_str}'

    print(f"Installing: {len(PIP_PACKAGES)} packages")
    result = run_command(cmd, check=False)

    if result is None:
        print("\n⚠️  Some packages may have failed to install")
        print("This is sometimes okay - we'll verify next")
    else:
        print("\n✓ Package installation complete")

    return True


def install_napari_plugin():
    """Install the manual crop napari plugin."""
    print("\nInstalling napari manual crop plugin...")

    # Plugin is located in the util_Scripts folder
    # Try new restructured path first, then current, then legacy
    plugin_dir = None
    for candidate in [
        Path(__file__).parent.parent / "src" / "mousebrain" / "pipeline_3d",  # mousebrain package
        Path(__file__).parent / "MouseBrain_Pipeline" / "3D_Cleared" / "util_Scripts" / "napari_manual_crop",
        Path(__file__).parent / "3D_Cleared" / "util_Scripts" / "napari_manual_crop",
        Path(__file__).parent / "3_Nuclei_Detection" / "util_Scripts" / "napari_manual_crop",
    ]:
        if candidate.exists():
            plugin_dir = candidate
            break
    if plugin_dir is None:
        plugin_dir = Path(__file__).parent / "MouseBrain_Pipeline" / "3D_Cleared" / "util_Scripts" / "napari_manual_crop"

    if not plugin_dir.exists():
        print(f"⚠️  Plugin directory not found: {plugin_dir}")
        print("   Plugin may need to be installed manually")
        return False

    print(f"✓ Plugin directory found: {plugin_dir}")

    # Install plugin in development mode
    if os.name == 'nt':
        pip_cmd = f"conda run -n {ENV_NAME} pip"
    else:
        pip_cmd = f"conda run -n {ENV_NAME} pip"

    if (plugin_dir / "setup.py").exists():
        cmd = f'{pip_cmd} install -e {plugin_dir}'
        result = run_command(cmd, check=False)
        if result is not None:
            print("✓ Plugin installed")
            return True

    print("⚠️  Plugin files not yet created - will be available after setup")
    return True


def verify_installation():
    """Verify that key packages are installed correctly."""
    print("\nVerifying installation...")

    if os.name == 'nt':
        python_cmd = f"conda run -n {ENV_NAME} python"
    else:
        python_cmd = f"conda run -n {ENV_NAME} python"

    checks = [
        ("brainreg", "import brainreg"),
        ("cellfinder", "import cellfinder"),
        ("napari", "import napari"),
        ("tifffile", "import tifffile"),
        ("numpy", "import numpy"),
    ]

    all_passed = True
    for name, import_cmd in checks:
        result = run_command(
            f'{python_cmd} -c "{import_cmd}"',
            check=False,
            capture_output=True
        )
        if result and result.returncode == 0:
            print(f"  ✓ {name}")
        else:
            print(f"  ❌ {name} - import failed")
            all_passed = False

    return all_passed


def create_config():
    """Create default configuration file."""
    config_dir = Path.home() / ".sci_connectome"
    config_dir.mkdir(exist_ok=True)

    config_file = config_dir / "config.json"

    if config_file.exists():
        print(f"✓ Config file already exists: {config_file}")
        return

    # Default configuration
    # Uses new restructured paths with fallback to current structure
    config = {
        "brainglobe_root": "Y:\\2_Connectome\\Tissue\\MouseBrain_Pipeline\\3D_Cleared\\1_Brains",
        "brainglobe_root_fallback": "Y:\\2_Connectome\\Tissue\\3D_Cleared\\1_Brains",
        "summary_data": "Y:\\2_Connectome\\Tissue\\MouseBrain_Pipeline\\3D_Cleared\\2_Data_Summary",
        "summary_data_fallback": "Y:\\2_Connectome\\Tissue\\3D_Cleared\\2_Data_Summary",
        "default_orientation": "iar",
        "camera_pixel_size": 6.5,
    }

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✓ Created config file: {config_file}")
    print("\nYou can edit this file to change default paths and settings")


# =============================================================================
# MAIN INSTALLATION
# =============================================================================

def main():
    print_header("SCI-Connectome-Tools Installer")
    print("\nThis will install the complete BrainGlobe pipeline environment.")
    print(f"Environment name: {ENV_NAME}")
    print(f"Python version: {PYTHON_VERSION}")
    print(f"\nPackages to install: {len(PIP_PACKAGES)}")

    response = input("\nProceed with installation? [Y/n]: ").strip().lower()
    if response and response != 'y':
        print("Installation cancelled.")
        return 1

    total_steps = 6
    current_step = 0

    # Step 1: Check conda
    current_step += 1
    print_step(current_step, total_steps, "Checking for conda")
    if not check_conda():
        return 1

    # Step 2: Create/update environment
    current_step += 1
    print_step(current_step, total_steps, "Creating conda environment")
    if not create_or_update_env():
        return 1

    # Step 3: Install packages
    current_step += 1
    print_step(current_step, total_steps, "Installing Python packages")
    if not install_packages():
        print("⚠️  Package installation encountered issues")

    # Step 4: Install napari plugin
    current_step += 1
    print_step(current_step, total_steps, "Installing napari plugin")
    install_napari_plugin()

    # Step 5: Create config
    current_step += 1
    print_step(current_step, total_steps, "Creating configuration")
    create_config()

    # Step 6: Verify
    current_step += 1
    print_step(current_step, total_steps, "Verifying installation")
    if verify_installation():
        print("\n✓ All checks passed!")
    else:
        print("\n⚠️  Some packages failed verification")
        print("You may need to install them manually")

    # Final instructions
    print_header("Installation Complete!")
    print("\n✓ Environment created successfully")
    print(f"✓ To use the pipeline, first activate the environment:\n")
    print(f"    conda activate {ENV_NAME}\n")
    print("✓ Then navigate to the scripts folder and run the pipeline:")
    print("    cd Tissue\\MouseBrain_Pipeline\\3D_Cleared\\util_Scripts")
    print("    python 1_organize_pipeline.py")
    print("\n✓ To launch napari with the manual crop plugin:")
    print("    napari")
    print("\nSee INSTALL.md for detailed usage instructions.")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Installation failed with error:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
