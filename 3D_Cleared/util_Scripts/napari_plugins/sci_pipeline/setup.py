"""
SCI-Connectome Pipeline Dashboard - napari plugin

Track brain processing progress and run pipeline steps from napari.

Install:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="sci-pipeline",
    version="0.1.0",
    author="Logan Friedrich",
    author_email="logan.friedrich@marquette.edu",
    description="Pipeline dashboard for BrainGlobe cell detection",
    packages=find_packages(),
    package_data={"sci_pipeline": ["napari.yaml"]},
    python_requires=">=3.8",
    install_requires=["napari", "qtpy"],
    entry_points={"napari.manifest": ["sci-pipeline = sci_pipeline"]},
)
