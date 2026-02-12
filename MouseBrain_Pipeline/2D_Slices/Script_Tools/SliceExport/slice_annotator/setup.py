"""
Slice Annotator - ND2 Annotation & Export Suite for napari.

A comprehensive napari plugin for working with ND2 microscopy files.
Provides annotation tools, multi-channel management, and publication-ready
TIFF export with scale bars and metadata overlays.

INSTALL (New Environment):
--------------------------
    cd y:\\2_Connectome\\5_Enhancer_Aim\\Script_Tools\\SliceExport\\slice_annotator
    conda env create -f environment.yml
    conda activate slice-annotator
    pip install -e .

INSTALL (Existing Environment):
-------------------------------
    conda activate your-env-name
    cd y:\\2_Connectome\\5_Enhancer_Aim\\Script_Tools\\SliceExport\\slice_annotator
    pip install -e .

USAGE:
------
    1. conda activate slice-annotator
    2. napari
    3. Plugins -> Slice Annotator
    4. Load ND2 file, adjust channels, annotate, export
"""

from setuptools import setup, find_packages

setup(
    name="slice-annotator",
    version="0.1.0",
    author="Logan Friedrich",
    author_email="logan.friedrich@marquette.edu",
    description="ND2 annotation and export tools for napari",
    long_description=__doc__,
    packages=find_packages(),
    package_data={
        "slice_annotator": ["napari.yaml"],
    },
    python_requires=">=3.8",
    install_requires=[
        "napari>=0.4.17",
        "numpy>=1.20",
        "dask>=2022.1",
        "nd2>=0.5.0",
        "tifffile>=2022.1",
        "qtpy>=2.0",
        "pillow>=9.0",
        "natsort>=8.0",
    ],
    entry_points={
        "napari.manifest": [
            "slice-annotator = slice_annotator:napari.yaml",
        ],
    },
    classifiers=[
        "Framework :: napari",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
