"""
SCI-Connectome Pipeline Tools for napari.

This is the MAIN plugin for running the BrainGlobe cell detection pipeline.
It provides GUI widgets so you don't need to use the command line.

WHAT'S INCLUDED:
----------------
1. Pipeline Dashboard  - See all your brains, their progress, run next step
2. Setup & Tuning      - Configure voxel size, orientation, detection params
3. Experiments         - Browse, search, and compare your experiment runs

INSTALL:
--------
    conda activate brainglobe-env
    cd y:\\2_Connectome\\3_Nuclei_Detection\\util_Scripts\\sci_connectome_napari
    pip install -e .

Or from repo root:
    cd y:\\2_Connectome
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="sci-connectome-pipeline",
    version="0.1.0",
    author="Logan Friedrich",
    author_email="logan.friedrich@marquette.edu",
    description="BrainGlobe pipeline GUI tools for napari",
    long_description=__doc__,
    packages=find_packages(),
    package_data={
        "sci_connectome_napari": ["napari.yaml"],
    },
    python_requires=">=3.8",
    install_requires=[
        "napari",
        "numpy",
        "tifffile",
        "qtpy",
    ],
    entry_points={
        "napari.manifest": [
            "sci-connectome-pipeline = sci_connectome_napari:napari.yaml",
        ],
    },
    classifiers=[
        "Framework :: napari",
        "Programming Language :: Python :: 3",
    ],
)
