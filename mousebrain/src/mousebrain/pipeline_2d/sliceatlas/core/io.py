"""
io.py - Image I/O for BrainSlice

Handles loading ND2 (Nikon) and TIFF files with metadata extraction.

Usage:
    from mousebrain.pipeline_2d.sliceatlas.core.io import load_image, extract_channels

    data, metadata = load_image("sample.nd2")
    red, green = extract_channels(data, red_idx=0, green_idx=1)
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List, Union
import numpy as np

# Lazy imports for optional dependencies
_nd2 = None
_tifffile = None
_aicsimageio = None


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


def _get_aicsimageio():
    """Lazy import aicsimageio for advanced format support."""
    global _aicsimageio
    if _aicsimageio is None:
        try:
            from aicsimageio import AICSImage
            _aicsimageio = AICSImage
        except ImportError:
            _aicsimageio = False  # Mark as unavailable
    return _aicsimageio


def load_image(
    file_path: Union[str, Path],
    z_projection: str = 'max',
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load ND2 or TIFF image with metadata.

    Args:
        file_path: Path to ND2 or TIFF file
        z_projection: How to handle Z dimension - 'max', 'mean', 'first', or 'all'

    Returns:
        Tuple of (image_array, metadata_dict)
        - image_array: shape (C, Y, X) for multi-channel 2D images
        - metadata: dict with voxel_size_um, channels, shape, dtype, etc.

    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == '.nd2':
        return _load_nd2(file_path, z_projection=z_projection)
    elif suffix in ['.tif', '.tiff']:
        return _load_tiff(file_path)
    else:
        # Try aicsimageio as fallback for other formats
        aics = _get_aicsimageio()
        if aics and aics is not False:
            return _load_aicsimageio(file_path)
        else:
            raise ValueError(
                f"Unsupported image format: {suffix}. "
                f"Supported formats: .nd2, .tif, .tiff"
            )


def _load_nd2(
    file_path: Path,
    z_projection: str = 'max',
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load Nikon ND2 file using nd2 library.

    Returns data in (C, Y, X) format for 2D multi-channel images.

    Args:
        file_path: Path to ND2 file
        z_projection: How to handle Z dimension - 'max', 'mean', 'first', or 'all'

    Returns:
        Tuple of (data, metadata)
    """
    nd2 = _get_nd2()

    with nd2.ND2File(file_path) as f:
        data = f.asarray()

        # Get dimension info from nd2 library
        sizes = f.sizes if hasattr(f, 'sizes') else {}

        # Extract metadata
        metadata = {
            'file_path': str(file_path),
            'file_format': 'nd2',
            'original_shape': data.shape,
            'dtype': str(data.dtype),
            'original_ndim': data.ndim,
            'sizes': dict(sizes),
        }

        # Try to get voxel sizes from nd2 voxel_size() method
        try:
            voxel_size = f.voxel_size()
            if voxel_size:
                metadata['voxel_size_um'] = {
                    'x': voxel_size.x if voxel_size.x else 1.0,
                    'y': voxel_size.y if voxel_size.y else 1.0,
                }
                if voxel_size.z:
                    metadata['voxel_size_um']['z'] = voxel_size.z
        except Exception:
            metadata['voxel_size_um'] = {'x': 1.0, 'y': 1.0}

        # Try to get channel names
        try:
            if hasattr(f, 'metadata') and f.metadata:
                if hasattr(f.metadata, 'channels'):
                    channels = []
                    for ch in f.metadata.channels:
                        if hasattr(ch, 'channel') and hasattr(ch.channel, 'name'):
                            channels.append(ch.channel.name)
                        else:
                            channels.append(f"Channel_{len(channels)}")
                    metadata['channels'] = channels
        except Exception:
            pass

        # Handle dimension ordering based on sizes dict
        # Common orders: (Z, C, Y, X), (T, Z, C, Y, X), (C, Y, X), etc.
        n_z = sizes.get('Z', 1)
        n_c = sizes.get('C', 1)
        n_y = sizes.get('Y', data.shape[-2] if data.ndim >= 2 else 1)
        n_x = sizes.get('X', data.shape[-1] if data.ndim >= 1 else 1)

        metadata['n_z'] = n_z
        metadata['n_channels'] = n_c

        # Reshape to (Z, C, Y, X) if we have proper size info
        if sizes and data.ndim == 4:
            # Data should already be in correct order from nd2 library
            # Typical: (Z, C, Y, X)
            if n_z > 1:
                # Need to collapse Z dimension
                if z_projection == 'max':
                    data = np.max(data, axis=0)  # (C, Y, X)
                    metadata['z_projection'] = 'max'
                elif z_projection == 'mean':
                    data = np.mean(data, axis=0).astype(data.dtype)
                    metadata['z_projection'] = 'mean'
                elif z_projection == 'first':
                    data = data[0]  # Take first Z
                    metadata['z_projection'] = 'first'
                elif z_projection == 'all':
                    # Return all Z planes - data stays as (Z, C, Y, X)
                    metadata['z_projection'] = 'all'
                else:
                    # Default to max projection
                    data = np.max(data, axis=0)
                    metadata['z_projection'] = 'max'
            else:
                # Only 1 Z plane, just squeeze
                data = data[0]  # (C, Y, X)
                metadata['z_projection'] = 'single_z'

        elif data.ndim == 3:
            # Could be (C, Y, X) or (Z, Y, X)
            if n_c > 1 and data.shape[0] == n_c:
                # Already (C, Y, X)
                pass
            elif n_z > 1 and data.shape[0] == n_z:
                # (Z, Y, X) - need projection and add channel dim
                if z_projection != 'all':
                    if z_projection == 'max':
                        data = np.max(data, axis=0)
                    elif z_projection == 'mean':
                        data = np.mean(data, axis=0).astype(data.dtype)
                    else:
                        data = data[0]
                data = data[np.newaxis, :, :]  # Add channel dimension
                metadata['z_projection'] = z_projection
            elif data.shape[-1] <= 4:
                # (Y, X, C) -> (C, Y, X)
                data = np.moveaxis(data, -1, 0)
            # else assume it's already (C, Y, X)

        elif data.ndim == 2:
            # Single channel 2D
            data = data[np.newaxis, :, :]

        # Update shape info
        metadata['shape'] = data.shape
        metadata['ndim'] = data.ndim

        if 'channels' not in metadata:
            if data.ndim >= 3:
                metadata['channels'] = [f"Channel_{i}" for i in range(data.shape[0])]
            else:
                metadata['channels'] = ['Channel_0']

    return data, metadata


def _load_tiff(file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load TIFF file with optional OME metadata.

    Returns data in (C, Y, X) format for 2D multi-channel images.
    """
    tifffile = _get_tifffile()

    with tifffile.TiffFile(file_path) as tif:
        data = tif.asarray()

        metadata = {
            'file_path': str(file_path),
            'file_format': 'tiff',
            'shape': data.shape,
            'dtype': str(data.dtype),
            'ndim': data.ndim,
            'voxel_size_um': {'x': 1.0, 'y': 1.0},
        }

        # Try to extract OME metadata
        if tif.ome_metadata:
            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(tif.ome_metadata)
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}

                # Find Pixels element
                pixels = root.find('.//ome:Pixels', ns)
                if pixels is not None:
                    # Physical sizes
                    px = pixels.get('PhysicalSizeX')
                    py = pixels.get('PhysicalSizeY')
                    if px:
                        metadata['voxel_size_um']['x'] = float(px)
                    if py:
                        metadata['voxel_size_um']['y'] = float(py)

                    # Channel names
                    channels = []
                    for ch in pixels.findall('ome:Channel', ns):
                        name = ch.get('Name') or ch.get('ID', f'Channel_{len(channels)}')
                        channels.append(name)
                    if channels:
                        metadata['channels'] = channels
            except Exception:
                pass

        # Also try ImageJ metadata
        if tif.imagej_metadata:
            try:
                ij_meta = tif.imagej_metadata
                if 'spacing' in ij_meta:
                    metadata['voxel_size_um']['z'] = ij_meta['spacing']
            except Exception:
                pass

    # Handle dimension order
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
        metadata['channels'] = metadata.get('channels', ['Channel_0'])
    elif data.ndim == 3:
        if data.shape[0] <= 4:
            pass  # Already (C, Y, X)
        elif data.shape[-1] <= 4:
            data = np.moveaxis(data, -1, 0)

    if 'channels' not in metadata:
        metadata['channels'] = [f"Channel_{i}" for i in range(data.shape[0])]

    return data, metadata


def _load_aicsimageio(file_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load image using aicsimageio (supports CZI, LIF, etc.).
    """
    AICSImage = _get_aicsimageio()
    if AICSImage is False:
        raise ImportError(
            "aicsimageio is required for this file format. "
            "Install with: pip install aicsimageio"
        )

    img = AICSImage(file_path)

    # Get 2D data (squeeze out singleton dimensions)
    data = img.get_image_data("CYX")  # Channel, Y, X

    metadata = {
        'file_path': str(file_path),
        'file_format': file_path.suffix.lower().lstrip('.'),
        'shape': data.shape,
        'dtype': str(data.dtype),
        'ndim': data.ndim,
        'voxel_size_um': {
            'x': img.physical_pixel_sizes.X or 1.0,
            'y': img.physical_pixel_sizes.Y or 1.0,
        },
        'channels': list(img.channel_names) if img.channel_names else [
            f"Channel_{i}" for i in range(data.shape[0])
        ],
    }

    return data, metadata


def extract_channels(
    data: np.ndarray,
    red_idx: int = 0,
    green_idx: int = 1,
    blue_idx: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    """
    Extract specific channels from multi-channel image.

    Args:
        data: Image array with shape (C, Y, X)
        red_idx: Index of red/nuclear channel
        green_idx: Index of green/signal channel
        blue_idx: Optional index of blue channel

    Returns:
        Tuple of 2D arrays (red, green) or (red, green, blue) if blue_idx provided
    """
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array (C, Y, X), got {data.ndim}D")

    n_channels = data.shape[0]

    if red_idx >= n_channels:
        raise ValueError(f"red_idx {red_idx} out of range for {n_channels} channels")
    if green_idx >= n_channels:
        raise ValueError(f"green_idx {green_idx} out of range for {n_channels} channels")

    red = data[red_idx]
    green = data[green_idx]

    if blue_idx is not None:
        if blue_idx >= n_channels:
            raise ValueError(f"blue_idx {blue_idx} out of range for {n_channels} channels")
        blue = data[blue_idx]
        return red, green, blue

    return red, green


def save_tiff(
    data: np.ndarray,
    file_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    compress: bool = True,
) -> Path:
    """
    Save array as TIFF file.

    Args:
        data: Image array to save
        file_path: Output path
        metadata: Optional metadata to include
        compress: Whether to use compression

    Returns:
        Path to saved file
    """
    tifffile = _get_tifffile()
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    compression = 'zlib' if compress else None

    # Add basic ImageJ metadata if 3D
    imagej = data.ndim >= 3

    tifffile.imwrite(
        file_path,
        data,
        compression=compression,
        imagej=imagej,
    )

    return file_path


def get_channel_info(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Get channel information from an image file without loading full data.

    Args:
        file_path: Path to image file

    Returns:
        List of dicts with channel info (name, index, etc.)
    """
    _, metadata = load_image(file_path)

    channels = metadata.get('channels', [])
    return [
        {'index': i, 'name': name}
        for i, name in enumerate(channels)
    ]


# =============================================================================
# FOLDER / STACK LOADING
# =============================================================================

def find_images_in_folder(
    folder_path: Union[str, Path],
    extensions: Optional[List[str]] = None,
    sort: bool = True,
) -> List[Path]:
    """
    Find all supported image files in a folder.

    Args:
        folder_path: Path to folder containing images
        extensions: List of extensions to include (default: ['.nd2', '.tif', '.tiff'])
        sort: Sort files naturally by name (handles S1, S2, ..., S10, S11 correctly)

    Returns:
        List of Path objects for found images
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    if extensions is None:
        extensions = ['.nd2', '.tif', '.tiff']

    # Normalize extensions
    extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions]

    # Find all matching files
    files = []
    for ext in extensions:
        files.extend(folder.glob(f'*{ext}'))

    if sort:
        # Natural sort to handle S1, S2, ..., S10 correctly
        import re
        def natural_key(path):
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', path.stem)]
        files = sorted(files, key=natural_key)
    else:
        files = sorted(files)

    return files


def load_folder(
    folder_path: Union[str, Path],
    extensions: Optional[List[str]] = None,
    z_projection: str = 'max',
    stack_axis: str = 'first',
    progress_callback: Optional[callable] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load all images from a folder as a single stacked array.

    Args:
        folder_path: Path to folder containing images
        extensions: List of extensions to include (default: ['.nd2', '.tif', '.tiff'])
        z_projection: How to handle Z dimension in individual files ('max', 'mean', 'first')
        stack_axis: Where to add the slice/file dimension ('first' -> (S, C, Y, X))
        progress_callback: Optional callback(current, total, filename) for progress updates

    Returns:
        Tuple of (stacked_data, metadata)
        - stacked_data: shape (N_slices, C, Y, X) if stack_axis='first'
        - metadata: combined metadata with 'files' list and per-slice info
    """
    files = find_images_in_folder(folder_path, extensions)

    if not files:
        raise FileNotFoundError(f"No supported images found in {folder_path}")

    n_files = len(files)

    # Report progress for first file
    if progress_callback:
        progress_callback(1, n_files, files[0].name)

    # Load first image to get shape info
    first_data, first_meta = load_image(files[0], z_projection=z_projection)

    n_channels = first_data.shape[0] if first_data.ndim >= 3 else 1
    height = first_data.shape[-2]
    width = first_data.shape[-1]

    # Pre-allocate stack array
    stack_shape = (n_files, n_channels, height, width)
    stack = np.zeros(stack_shape, dtype=first_data.dtype)
    stack[0] = first_data

    # Track per-file metadata
    file_metadata = [{
        'file': str(files[0]),
        'index': 0,
        **first_meta,
    }]

    # Load remaining files
    for i, file_path in enumerate(files[1:], start=1):
        if progress_callback:
            progress_callback(i + 1, n_files, file_path.name)

        try:
            data, meta = load_image(file_path, z_projection=z_projection)

            # Handle shape mismatches
            if data.shape != first_data.shape:
                # Try to fit into stack (crop or pad)
                c = min(data.shape[0], n_channels)
                h = min(data.shape[-2], height)
                w = min(data.shape[-1], width)
                stack[i, :c, :h, :w] = data[:c, :h, :w]
            else:
                stack[i] = data

            file_metadata.append({
                'file': str(file_path),
                'index': i,
                **meta,
            })

        except Exception as e:
            print(f"Warning: Failed to load {file_path.name}: {e}")
            file_metadata.append({
                'file': str(file_path),
                'index': i,
                'error': str(e),
            })

    # Combined metadata
    combined_meta = {
        'folder': str(folder_path),
        'n_slices': n_files,
        'n_channels': n_channels,
        'shape': stack.shape,
        'dtype': str(stack.dtype),
        'stack_axis': stack_axis,
        'files': [str(f) for f in files],
        'file_metadata': file_metadata,
        'channels': first_meta.get('channels', [f'Channel_{i}' for i in range(n_channels)]),
        'voxel_size_um': first_meta.get('voxel_size_um', {'x': 1.0, 'y': 1.0}),
    }

    return stack, combined_meta


def load_folder_lazy(
    folder_path: Union[str, Path],
    extensions: Optional[List[str]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load folder as a lazy dask array (memory efficient for large datasets).

    Args:
        folder_path: Path to folder containing images
        extensions: List of extensions to include

    Returns:
        Tuple of (dask_array, metadata)
    """
    try:
        import dask
        import dask.array as da
    except ImportError:
        raise ImportError("dask is required for lazy loading. Install with: pip install dask")

    files = find_images_in_folder(folder_path, extensions)
    if not files:
        raise FileNotFoundError(f"No supported images found in {folder_path}")

    # Get shape from first file
    first_data, first_meta = load_image(files[0])
    shape_per_file = first_data.shape
    dtype = first_data.dtype

    # Create delayed loading functions
    def load_single(path):
        data, _ = load_image(path)
        return data

    # Build lazy stack
    lazy_arrays = [
        da.from_delayed(
            dask.delayed(load_single)(f),
            shape=shape_per_file,
            dtype=dtype
        )
        for f in files
    ]

    stack = da.stack(lazy_arrays, axis=0)

    metadata = {
        'folder': str(folder_path),
        'n_slices': len(files),
        'shape': stack.shape,
        'dtype': str(dtype),
        'files': [str(f) for f in files],
        'channels': first_meta.get('channels', []),
        'voxel_size_um': first_meta.get('voxel_size_um', {'x': 1.0, 'y': 1.0}),
        'lazy': True,
    }

    return stack, metadata


def guess_channel_roles(
    metadata: Dict[str, Any]
) -> Dict[str, int]:
    """
    Guess which channels are nuclear (red) and signal (green) based on names.

    Handles common naming patterns:
    - Text names: 'DAPI', 'GFP', 'mScarlet', 'eYFP', etc.
    - Wavelength names: '405', '488', '561', '640', etc.

    Args:
        metadata: Metadata dict from load_image

    Returns:
        Dict with 'nuclear' and 'signal' keys mapped to channel indices
    """
    channels = metadata.get('channels', [])

    # Default to first two channels
    n_channels = len(channels)
    roles = {
        'nuclear': 1 if n_channels > 1 else 0,  # Red typically second
        'signal': 0,  # Green typically first
    }

    # Keywords for nuclear channels (red/far-red fluorophores)
    nuclear_keywords = [
        'dapi', 'hoechst', 'nuclear', 'nuclei', 'red',
        'mscarlet', 'scarlet', 'rfp', 'mcherry', 'tdtomato',
        'h2b',  # H2B-mScarlet
    ]

    # Keywords for signal channels (green/yellow fluorophores)
    signal_keywords = [
        'gfp', 'green', 'yfp', 'eyfp', 'signal',
        'fitc', 'alexa488', 'egfp', 'venus',
    ]

    # Wavelength mappings (excitation wavelengths in nm)
    # Red/far-red wavelengths (nuclear markers)
    red_wavelengths = ['561', '568', '594', '633', '640', '647', '660']
    # Green wavelengths (signal markers)
    green_wavelengths = ['488', '491', '509', '514']
    # Blue wavelengths (could be nuclear with DAPI)
    blue_wavelengths = ['405', '440', '458']

    for i, name in enumerate(channels):
        name_lower = name.lower().strip()

        # Check text keywords first
        for kw in nuclear_keywords:
            if kw in name_lower:
                roles['nuclear'] = i
                break

        for kw in signal_keywords:
            if kw in name_lower:
                roles['signal'] = i
                break

        # Check wavelength patterns
        # Extract numbers from channel name
        numbers = ''.join(c for c in name if c.isdigit())
        if numbers:
            if numbers in red_wavelengths:
                roles['nuclear'] = i
            elif numbers in green_wavelengths:
                roles['signal'] = i
            elif numbers in blue_wavelengths:
                # Blue typically used for DAPI/nuclear
                roles['nuclear'] = i

    return roles
