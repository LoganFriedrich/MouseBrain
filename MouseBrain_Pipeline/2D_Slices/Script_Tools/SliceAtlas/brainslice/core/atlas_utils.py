"""
atlas_utils.py - Brain and spinal cord atlas utilities for BrainSlice

Provides access to Allen Brain Atlas and Allen Spinal Cord Atlas
via BrainGlobe Atlas API.

Usage:
    from brainslice.core.atlas_utils import DualAtlasManager

    manager = DualAtlasManager()
    slice_img = manager.get_reference_slice("brain", position_um=5000, orientation="coronal")
    region_name = manager.get_region_name(region_id=672, atlas_type="brain")
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np

# Lazy import for brainglobe
_bg_atlas = None


def _get_brainglobe():
    """Lazy import BrainGlobe Atlas API."""
    global _bg_atlas
    if _bg_atlas is None:
        try:
            from brainglobe_atlasapi import BrainGlobeAtlas
            _bg_atlas = BrainGlobeAtlas
        except ImportError:
            raise ImportError(
                "brainglobe-atlasapi is required for atlas access. "
                "Install with: pip install brainglobe-atlasapi"
            )
    return _bg_atlas


# Available atlases
BRAIN_ATLASES = {
    'allen_mouse_10um': 'Allen Mouse Brain Atlas at 10um resolution',
    'allen_mouse_25um': 'Allen Mouse Brain Atlas at 25um resolution',
    'allen_mouse_50um': 'Allen Mouse Brain Atlas at 50um resolution',
}

SPINAL_CORD_ATLASES = {
    'allen_cord_20um': 'Allen Mouse Spinal Cord Atlas (20x10x10um)',
}

ALL_ATLASES = {**BRAIN_ATLASES, **SPINAL_CORD_ATLASES}


class DualAtlasManager:
    """
    Manages access to both brain and spinal cord atlases.

    Provides unified interface for:
    - Getting reference slices at specific positions
    - Looking up region names from IDs
    - Getting region hierarchy information
    """

    def __init__(
        self,
        brain_atlas: str = 'allen_mouse_10um',
        spinal_atlas: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize atlas manager.

        Args:
            brain_atlas: Brain atlas name (default: allen_mouse_10um)
            spinal_atlas: Spinal cord atlas name (default: None, loaded on demand)
            cache_dir: Optional custom cache directory for atlas files
        """
        BrainGlobeAtlas = _get_brainglobe()

        self.brain_atlas_name = brain_atlas
        self.spinal_atlas_name = spinal_atlas

        # Load brain atlas
        self._brain_atlas = BrainGlobeAtlas(brain_atlas)

        # Spinal cord atlas loaded on demand
        self._spinal_atlas = None
        if spinal_atlas:
            try:
                self._spinal_atlas = BrainGlobeAtlas(spinal_atlas)
            except Exception as e:
                print(f"Warning: Could not load spinal cord atlas: {e}")

    @property
    def brain_atlas(self):
        """Get brain atlas object."""
        return self._brain_atlas

    @property
    def spinal_atlas(self):
        """Get spinal cord atlas object (loads on demand)."""
        if self._spinal_atlas is None and self.spinal_atlas_name:
            BrainGlobeAtlas = _get_brainglobe()
            self._spinal_atlas = BrainGlobeAtlas(self.spinal_atlas_name)
        return self._spinal_atlas

    def get_atlas(self, atlas_type: str = 'brain'):
        """
        Get atlas object by type.

        Args:
            atlas_type: 'brain' or 'spinal_cord'

        Returns:
            BrainGlobeAtlas object
        """
        if atlas_type == 'brain':
            return self._brain_atlas
        elif atlas_type in ['spinal_cord', 'spinal']:
            if self._spinal_atlas is None:
                raise ValueError(
                    "Spinal cord atlas not loaded. "
                    "Initialize with spinal_atlas parameter or call load_spinal_atlas()"
                )
            return self._spinal_atlas
        else:
            raise ValueError(f"Unknown atlas type: {atlas_type}")

    def load_spinal_atlas(self, atlas_name: str = 'allen_cord_20um'):
        """Load spinal cord atlas on demand."""
        BrainGlobeAtlas = _get_brainglobe()
        self.spinal_atlas_name = atlas_name
        self._spinal_atlas = BrainGlobeAtlas(atlas_name)

    def get_atlas_info(self, atlas_type: str = 'brain') -> Dict[str, Any]:
        """
        Get information about an atlas.

        Returns:
            Dict with atlas name, resolution, shape, etc.
        """
        atlas = self.get_atlas(atlas_type)

        return {
            'name': atlas.atlas_name,
            'resolution': atlas.resolution,
            'shape': atlas.shape,
            'orientation': atlas.orientation,
            'n_regions': len(atlas.structures),
        }

    def get_reference_slice(
        self,
        atlas_type: str = 'brain',
        position_um: Optional[float] = None,
        position_idx: Optional[int] = None,
        orientation: str = 'coronal',
    ) -> np.ndarray:
        """
        Get a 2D reference slice from the atlas at a specific position.

        Args:
            atlas_type: 'brain' or 'spinal_cord'
            position_um: Position in micrometers (converted to index)
            position_idx: Direct slice index (if position_um not provided)
            orientation: 'coronal', 'sagittal', or 'horizontal'

        Returns:
            2D reference image array
        """
        atlas = self.get_atlas(atlas_type)

        # Get reference volume
        reference = atlas.reference

        # Determine axis based on orientation
        if orientation == 'coronal':
            axis = 0  # Anterior-posterior
        elif orientation == 'sagittal':
            axis = 2  # Left-right
        elif orientation == 'horizontal':
            axis = 1  # Dorsal-ventral
        else:
            raise ValueError(f"Unknown orientation: {orientation}")

        # Convert position to index
        if position_idx is not None:
            idx = position_idx
        elif position_um is not None:
            resolution = atlas.resolution[axis]
            idx = int(position_um / resolution)
        else:
            # Default to middle
            idx = reference.shape[axis] // 2

        # Clamp to valid range
        idx = max(0, min(idx, reference.shape[axis] - 1))

        # Extract slice
        if axis == 0:
            slice_img = reference[idx, :, :]
        elif axis == 1:
            slice_img = reference[:, idx, :]
        else:
            slice_img = reference[:, :, idx]

        return slice_img

    def get_annotation_slice(
        self,
        atlas_type: str = 'brain',
        position_um: Optional[float] = None,
        position_idx: Optional[int] = None,
        orientation: str = 'coronal',
    ) -> np.ndarray:
        """
        Get a 2D annotation (region labels) slice from the atlas.

        Returns:
            2D array where each pixel value is a region ID
        """
        atlas = self.get_atlas(atlas_type)
        annotation = atlas.annotation

        # Same logic as reference slice
        if orientation == 'coronal':
            axis = 0
        elif orientation == 'sagittal':
            axis = 2
        elif orientation == 'horizontal':
            axis = 1
        else:
            raise ValueError(f"Unknown orientation: {orientation}")

        if position_idx is not None:
            idx = position_idx
        elif position_um is not None:
            resolution = atlas.resolution[axis]
            idx = int(position_um / resolution)
        else:
            idx = annotation.shape[axis] // 2

        idx = max(0, min(idx, annotation.shape[axis] - 1))

        if axis == 0:
            return annotation[idx, :, :]
        elif axis == 1:
            return annotation[:, idx, :]
        else:
            return annotation[:, :, idx]

    def get_region_name(
        self,
        region_id: int,
        atlas_type: str = 'brain',
    ) -> str:
        """
        Get human-readable region name from ID.

        Args:
            region_id: Atlas region ID
            atlas_type: 'brain' or 'spinal_cord'

        Returns:
            Region name string, or 'Unknown' if not found
        """
        if region_id == 0:
            return 'Outside Brain'

        atlas = self.get_atlas(atlas_type)

        try:
            structure = atlas.structures[region_id]
            return structure['name']
        except (KeyError, IndexError):
            return f'Unknown ({region_id})'

    def get_region_info(
        self,
        region_id: int,
        atlas_type: str = 'brain',
    ) -> Dict[str, Any]:
        """
        Get full information about a region.

        Returns:
            Dict with name, acronym, parent, children, etc.
        """
        if region_id == 0:
            return {
                'id': 0,
                'name': 'Outside Brain',
                'acronym': 'OB',
                'parent': None,
            }

        atlas = self.get_atlas(atlas_type)

        try:
            structure = atlas.structures[region_id]
            return {
                'id': region_id,
                'name': structure.get('name', 'Unknown'),
                'acronym': structure.get('acronym', ''),
                'parent': structure.get('parent_structure_id'),
                'rgb_triplet': structure.get('rgb_triplet', [128, 128, 128]),
            }
        except (KeyError, IndexError):
            return {
                'id': region_id,
                'name': f'Unknown ({region_id})',
                'acronym': f'UNK{region_id}',
                'parent': None,
            }

    def get_all_regions(
        self,
        atlas_type: str = 'brain',
    ) -> List[Dict[str, Any]]:
        """
        Get list of all regions in the atlas.

        Returns:
            List of region info dicts
        """
        atlas = self.get_atlas(atlas_type)

        regions = []
        for structure_id, structure in atlas.structures.items():
            regions.append({
                'id': structure_id,
                'name': structure.get('name', 'Unknown'),
                'acronym': structure.get('acronym', ''),
            })

        return regions

    def get_parent_region(
        self,
        region_id: int,
        atlas_type: str = 'brain',
        level: int = 1,
    ) -> Optional[int]:
        """
        Get parent region ID at a specific level up the hierarchy.

        Args:
            region_id: Starting region ID
            atlas_type: Atlas type
            level: Number of levels to go up (1 = immediate parent)

        Returns:
            Parent region ID, or None if at top level
        """
        atlas = self.get_atlas(atlas_type)

        current_id = region_id
        for _ in range(level):
            try:
                structure = atlas.structures[current_id]
                parent_id = structure.get('parent_structure_id')
                if parent_id is None or parent_id == 0:
                    return None
                current_id = parent_id
            except (KeyError, IndexError):
                return None

        return current_id

    def is_brain_region(self, region_name: str) -> bool:
        """
        Check if a region name belongs to brain (vs spinal cord).

        Uses heuristics based on region names.
        """
        brain_keywords = [
            'cortex', 'hippocampus', 'thalamus', 'hypothalamus',
            'cerebellum', 'brainstem', 'midbrain', 'pons', 'medulla',
            'striatum', 'amygdala', 'olfactory', 'fiber tracts'
        ]
        spinal_keywords = [
            'spinal', 'cervical', 'thoracic', 'lumbar', 'sacral',
            'cord', 'dorsal horn', 'ventral horn'
        ]

        name_lower = region_name.lower()

        for kw in spinal_keywords:
            if kw in name_lower:
                return False

        for kw in brain_keywords:
            if kw in name_lower:
                return True

        # Default to brain
        return True


def list_available_atlases() -> Dict[str, str]:
    """
    List all available atlases that can be loaded.

    Returns:
        Dict mapping atlas name to description
    """
    return ALL_ATLASES.copy()


def download_atlas(atlas_name: str) -> bool:
    """
    Download an atlas if not already cached.

    Args:
        atlas_name: Name of atlas to download

    Returns:
        True if successful
    """
    BrainGlobeAtlas = _get_brainglobe()

    try:
        # Creating the atlas will download if needed
        _ = BrainGlobeAtlas(atlas_name)
        return True
    except Exception as e:
        print(f"Error downloading atlas {atlas_name}: {e}")
        return False
