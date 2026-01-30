"""
io.py - ND2 and TIFF I/O for Slice Annotator.

Handles loading ND2 files with full z-stack support, lazy loading via dask,
and TIFF export with metadata.

Usage:
    from braintools.pipeline_2d.annotator.core.io import load_nd2, load_nd2_lazy, save_tiff

    # Load full data
    data, metadata = load_nd2("sample.nd2")

    # Lazy load for large files
    data, metadata = load_nd2_lazy("large_sample.nd2")

    # Get maximum intensity projection
    mip = get_mip(data)
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List, Union
import numpy as np

# Lazy imports for optional dependencies
_nd2 = None
_tifffile = None
_dask = None


def _get_nd2():
    """Lazy import nd2 library."""
    global _nd2
    if _nd2 is None:
        try:
            import nd2
            _nd2 = nd2
        except ImportError:
            raise ImportError(
                "nd2 library is required for ND2 file support. "
                "Install with: pip install nd2"
            )
    return _nd2


def _get_tifffile():
    """Lazy import tifffile library."""
    global _tifffile
    if _tifffile is None:
        import tifffile
        _tifffile = tifffile
    return _tifffile


def _get_dask():
    """Lazy import dask library."""
    global _dask
    if _dask is None:
        try:
            import dask
            import dask.array as da
            _dask = (dask, da)
        except ImportError:
            raise ImportError(
                "dask is required for lazy loading. "
                "Install with: pip install dask"
            )
    return _dask


def load_nd2(
    file_path: Union[str, Path],
    z_projection: Optional[str] = 'max',
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load ND2 file with MIP (Maximum Intensity Projection) applied by default.

    Each ND2 file contains an optical z-stack from confocal imaging. By default,
    this function applies MIP to flatten the optical z-stack into a single 2D
    image per channel.

    Args:
        file_path: Path to ND2 file
        z_projection: Projection mode - 'max' (default), 'mean', 'min', 'sum',
                      or None to keep full optical z-stack

    Returns:
        Tuple of (data, metadata)
        - data: numpy array with shape (C, Y, X) if projected, or (Z, C, Y, X) if z_projection=None
        - metadata: dict with voxel_size_um, channel_names, shape, z_projection applied, etc.
    """
    nd2 = _get_nd2()
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"ND2 file not found: {file_path}")

    with nd2.ND2File(file_path) as f:
        data = f.asarray()

        # Get dimension info
        sizes = f.sizes if hasattr(f, 'sizes') else {}

        # Extract metadata
        metadata = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_format': 'nd2',
            'original_shape': data.shape,
            'dtype': str(data.dtype),
            'sizes': dict(sizes),
        }

        # Extract voxel sizes
        try:
            voxel_size = f.voxel_size()
            if voxel_size:
                metadata['voxel_size_um'] = {
                    'x': voxel_size.x if voxel_size.x else 1.0,
                    'y': voxel_size.y if voxel_size.y else 1.0,
                    'z': voxel_size.z if voxel_size.z else 1.0,
                }
        except Exception:
            metadata['voxel_size_um'] = {'x': 1.0, 'y': 1.0, 'z': 1.0}

        # Extract channel names
        channel_names = []
        try:
            if hasattr(f, 'metadata') and f.metadata:
                if hasattr(f.metadata, 'channels'):
                    for ch in f.metadata.channels:
                        if hasattr(ch, 'channel') and hasattr(ch.channel, 'name'):
                            channel_names.append(ch.channel.name)
                        else:
                            channel_names.append(f"Channel_{len(channel_names)}")
        except Exception:
            pass

        # Get dimension counts
        n_z = sizes.get('Z', 1)
        n_c = sizes.get('C', 1)
        n_t = sizes.get('T', 1)  # Time dimension if present

        metadata['n_z'] = n_z
        metadata['n_channels'] = n_c
        metadata['n_timepoints'] = n_t

        # Generate slice names (Z indices)
        metadata['slice_names'] = [f"Z{i:04d}" for i in range(n_z)]

        # Normalize dimension order to (Z, C, Y, X) or (T, Z, C, Y, X)
        # ND2 files commonly have dimensions in order specified by sizes dict
        data = _normalize_dimensions(data, sizes, n_t, n_z, n_c)

        # Handle T dimension if present (take first timepoint for now)
        if n_t > 1 and data.ndim == 5:
            data = data[0]  # Take first timepoint, now (Z, C, Y, X)
            metadata['timepoint_selected'] = 0

        # Apply z-projection if requested, or squeeze singleton Z dimension
        if z_projection is not None:
            if n_z > 1:
                # Apply actual projection (MIP, mean, etc.)
                data = get_mip(data, projection=z_projection)
                metadata['z_projection'] = z_projection
            elif data.ndim == 4 and data.shape[0] == 1:
                # Squeeze singleton Z dimension: (1, C, Y, X) -> (C, Y, X)
                data = data[0]
                metadata['z_projection'] = 'squeeze'
            else:
                metadata['z_projection'] = None
        else:
            metadata['z_projection'] = None

        # Ensure we have proper channel names
        if not channel_names:
            actual_n_c = data.shape[-3] if data.ndim >= 3 else 1
            channel_names = [f"Channel_{i}" for i in range(actual_n_c)]
        metadata['channel_names'] = channel_names

        # Update shape info
        metadata['shape'] = data.shape
        metadata['ndim'] = data.ndim

    return data, metadata


def _normalize_dimensions(
    data: np.ndarray,
    sizes: Dict[str, int],
    n_t: int,
    n_z: int,
    n_c: int,
) -> np.ndarray:
    """
    Normalize array dimensions to standard order.

    Target: (Z, C, Y, X) or (T, Z, C, Y, X) if time present
    """
    # Get dimension order from sizes dict
    dim_order = list(sizes.keys()) if sizes else []

    # Common dimension patterns and how to handle them
    if data.ndim == 4:
        # Could be (Z, C, Y, X), (T, C, Y, X), (C, Z, Y, X), etc.
        if n_z > 1 and n_c > 1:
            # Check if first dim is Z or C
            if data.shape[0] == n_z and data.shape[1] == n_c:
                pass  # Already (Z, C, Y, X)
            elif data.shape[0] == n_c and data.shape[1] == n_z:
                # (C, Z, Y, X) -> (Z, C, Y, X)
                data = np.moveaxis(data, 0, 1)
        elif n_z > 1 and n_c == 1:
            # (Z, Y, X) with implicit channel
            if data.ndim == 3:
                data = data[:, np.newaxis, :, :]  # Add channel dim

    elif data.ndim == 3:
        # Could be (C, Y, X), (Z, Y, X), or (Y, X, C)
        if data.shape[-1] <= 4 and data.shape[-1] < data.shape[-2]:
            # Likely (Y, X, C), transpose to (C, Y, X)
            data = np.moveaxis(data, -1, 0)

        if n_z > 1 and data.shape[0] == n_z:
            # (Z, Y, X) - add channel dimension
            data = data[:, np.newaxis, :, :]  # -> (Z, C, Y, X)
        elif n_c > 1 or data.shape[0] <= 8:
            # (C, Y, X) - add Z dimension of 1
            data = data[np.newaxis, :, :, :]  # -> (Z, C, Y, X)

    elif data.ndim == 2:
        # Single 2D image
        data = data[np.newaxis, np.newaxis, :, :]  # -> (Z, C, Y, X)

    elif data.ndim == 5:
        # (T, Z, C, Y, X) or similar - keep as is for now
        pass

    return data


def load_nd2_lazy(
    file_path: Union[str, Path],
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load ND2 file as lazy dask array for memory-efficient handling.

    Args:
        file_path: Path to ND2 file

    Returns:
        Tuple of (dask_array, metadata)
        - dask_array: lazy dask array with shape (Z, C, Y, X)
        - metadata: dict with file info and voxel sizes
    """
    nd2 = _get_nd2()
    dask, da = _get_dask()
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"ND2 file not found: {file_path}")

    with nd2.ND2File(file_path) as f:
        # Get lazy dask array if available
        if hasattr(f, 'to_dask'):
            data = f.to_dask()
        else:
            # Fall back to regular load
            data = da.from_array(f.asarray(), chunks='auto')

        sizes = f.sizes if hasattr(f, 'sizes') else {}

        # Build metadata (same as load_nd2)
        metadata = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_format': 'nd2',
            'shape': data.shape,
            'dtype': str(data.dtype),
            'sizes': dict(sizes),
            'lazy': True,
        }

        # Extract voxel sizes
        try:
            voxel_size = f.voxel_size()
            if voxel_size:
                metadata['voxel_size_um'] = {
                    'x': voxel_size.x if voxel_size.x else 1.0,
                    'y': voxel_size.y if voxel_size.y else 1.0,
                    'z': voxel_size.z if voxel_size.z else 1.0,
                }
        except Exception:
            metadata['voxel_size_um'] = {'x': 1.0, 'y': 1.0, 'z': 1.0}

        # Extract channel names
        channel_names = []
        try:
            if hasattr(f, 'metadata') and f.metadata:
                if hasattr(f.metadata, 'channels'):
                    for ch in f.metadata.channels:
                        if hasattr(ch, 'channel') and hasattr(ch.channel, 'name'):
                            channel_names.append(ch.channel.name)
                        else:
                            channel_names.append(f"Channel_{len(channel_names)}")
        except Exception:
            pass

        n_z = sizes.get('Z', 1)
        n_c = sizes.get('C', 1)

        if not channel_names:
            channel_names = [f"Channel_{i}" for i in range(n_c)]

        metadata['channel_names'] = channel_names
        metadata['n_z'] = n_z
        metadata['n_channels'] = n_c
        metadata['slice_names'] = [f"Z{i:04d}" for i in range(n_z)]

    return data, metadata


def get_mip(
    data: np.ndarray,
    axis: int = 0,
    projection: str = 'max',
) -> np.ndarray:
    """
    Compute intensity projection along specified axis.

    Args:
        data: Input array, typically (Z, C, Y, X)
        axis: Axis to project along (default 0 for Z)
        projection: Projection type - 'max', 'mean', 'min', 'sum'

    Returns:
        Projected array with one fewer dimension
    """
    if projection == 'max':
        return np.max(data, axis=axis)
    elif projection == 'mean':
        return np.mean(data, axis=axis).astype(data.dtype)
    elif projection == 'min':
        return np.min(data, axis=axis)
    elif projection == 'sum':
        return np.sum(data, axis=axis)
    else:
        raise ValueError(f"Unknown projection type: {projection}")


def save_tiff(
    data: np.ndarray,
    file_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    compress: bool = True,
    imagej: bool = True,
) -> Path:
    """
    Save array as TIFF file.

    Args:
        data: Image array to save
        file_path: Output path
        metadata: Optional metadata (voxel_size_um will be used for resolution)
        compress: Whether to use compression
        imagej: Write ImageJ-compatible TIFF

    Returns:
        Path to saved file
    """
    tifffile = _get_tifffile()
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    compression = 'zlib' if compress else None

    # Build resolution info from metadata
    resolution = None
    resolution_unit = None
    if metadata and 'voxel_size_um' in metadata:
        voxel = metadata['voxel_size_um']
        # Resolution is pixels per unit, TIFF uses centimeters
        # Convert microns to cm: 1 um = 0.0001 cm
        if voxel.get('x') and voxel.get('y'):
            px_per_cm_x = 10000.0 / voxel['x']  # pixels per cm
            px_per_cm_y = 10000.0 / voxel['y']
            resolution = (px_per_cm_x, px_per_cm_y)
            resolution_unit = 3  # RESUNIT_CENTIMETER

    tifffile.imwrite(
        file_path,
        data,
        compression=compression,
        imagej=imagej,
        resolution=resolution,
        resolutionunit=resolution_unit,
    )

    return file_path


def load_tiff(
    file_path: Union[str, Path],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load TIFF file with metadata extraction.

    Args:
        file_path: Path to TIFF file

    Returns:
        Tuple of (data, metadata)
    """
    tifffile = _get_tifffile()
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"TIFF file not found: {file_path}")

    with tifffile.TiffFile(file_path) as tif:
        data = tif.asarray()

        metadata = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_format': 'tiff',
            'shape': data.shape,
            'dtype': str(data.dtype),
            'ndim': data.ndim,
            'voxel_size_um': {'x': 1.0, 'y': 1.0, 'z': 1.0},
        }

        # Try to extract resolution from TIFF tags
        try:
            page = tif.pages[0]
            if page.tags.get('XResolution') and page.tags.get('YResolution'):
                x_res = page.tags['XResolution'].value
                y_res = page.tags['YResolution'].value
                unit = page.tags.get('ResolutionUnit')

                # Convert to microns
                if isinstance(x_res, tuple):
                    x_res = x_res[0] / x_res[1]
                if isinstance(y_res, tuple):
                    y_res = y_res[0] / y_res[1]

                # Resolution is pixels per unit
                if unit and unit.value == 3:  # Centimeter
                    metadata['voxel_size_um']['x'] = 10000.0 / x_res
                    metadata['voxel_size_um']['y'] = 10000.0 / y_res
        except Exception:
            pass

        # Try ImageJ metadata for Z spacing
        if tif.imagej_metadata:
            try:
                ij_meta = tif.imagej_metadata
                if 'spacing' in ij_meta:
                    metadata['voxel_size_um']['z'] = ij_meta['spacing']
            except Exception:
                pass

    return data, metadata


def load_folder(
    folder_path: Union[str, Path],
    z_projection: str = 'max',
    progress_callback: Optional[callable] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load folder of ND2 files as stacked tissue sections.

    Each ND2 file represents one physical tissue section (vibratome slice).
    This function:
    1. Finds all ND2 files in the folder (naturally sorted)
    2. Applies MIP to each file to flatten optical z-stacks
    3. Stacks all results into (N_tissue_slices, C, Y, X)

    The resulting array can be navigated with napari's z-slider, where each
    position represents a different tissue section (physical slice), NOT
    optical z-planes within one file.

    Args:
        folder_path: Path to folder containing ND2 files
        z_projection: Projection mode for each file - 'max' (default), 'mean', 'min', 'sum'
        progress_callback: Optional callback(current, total, filename) for progress updates

    Returns:
        Tuple of (stacked_data, metadata)
        - stacked_data: numpy array with shape (N_tissue_slices, C, Y, X)
        - metadata: dict with file list, channel names, voxel sizes, etc.
    """
    files = find_nd2_files(folder_path)

    if not files:
        raise FileNotFoundError(f"No ND2 files found in {folder_path}")

    n_files = len(files)

    # Progress for first file
    if progress_callback:
        progress_callback(1, n_files, files[0].name)

    # Load first file with MIP to get shape
    first_data, first_meta = load_nd2(files[0], z_projection=z_projection)

    # After MIP, shape should be (C, Y, X)
    if first_data.ndim != 3:
        raise ValueError(
            f"Expected 3D array (C, Y, X) after MIP, got {first_data.ndim}D. "
            f"Shape: {first_data.shape}"
        )

    n_channels, height, width = first_data.shape

    # Pre-allocate stack: (N_tissue_slices, C, Y, X)
    stack = np.zeros((n_files, n_channels, height, width), dtype=first_data.dtype)
    stack[0] = first_data

    # Track per-file info
    file_names = [files[0].name]

    # Load remaining files
    for i, file_path in enumerate(files[1:], start=1):
        if progress_callback:
            progress_callback(i + 1, n_files, file_path.name)

        try:
            data, _ = load_nd2(file_path, z_projection=z_projection)

            # Handle shape mismatches (crop to fit)
            if data.shape != first_data.shape:
                c = min(data.shape[0], n_channels)
                h = min(data.shape[-2], height)
                w = min(data.shape[-1], width)
                stack[i, :c, :h, :w] = data[:c, :h, :w]
            else:
                stack[i] = data

            file_names.append(file_path.name)

        except Exception as e:
            print(f"Warning: Failed to load {file_path.name}: {e}")
            file_names.append(f"{file_path.name} (FAILED)")

    # Build combined metadata
    metadata = {
        'folder': str(folder_path),
        'n_slices': n_files,
        'n_channels': n_channels,
        'shape': stack.shape,
        'dtype': str(stack.dtype),
        'z_projection': z_projection,
        'files': [str(f) for f in files],
        'file_names': file_names,
        'channel_names': first_meta.get('channel_names', [f'Channel_{i}' for i in range(n_channels)]),
        'voxel_size_um': first_meta.get('voxel_size_um', {'x': 1.0, 'y': 1.0}),
    }

    return stack, metadata


def find_nd2_files(
    folder_path: Union[str, Path],
    recursive: bool = False,
) -> List[Path]:
    """
    Find all ND2 files in a folder.

    Args:
        folder_path: Path to folder
        recursive: Search subdirectories

    Returns:
        List of Path objects for found ND2 files (naturally sorted)
    """
    from natsort import natsorted

    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    pattern = '**/*.nd2' if recursive else '*.nd2'
    files = list(folder.glob(pattern))

    # Natural sort
    return natsorted(files, key=lambda p: p.name)
