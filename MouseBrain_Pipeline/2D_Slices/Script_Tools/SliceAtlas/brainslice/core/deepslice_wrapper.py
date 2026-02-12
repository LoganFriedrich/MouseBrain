"""
deepslice_wrapper.py - DeepSlice integration for automatic atlas position detection

DeepSlice is a deep learning model that predicts CCF (Common Coordinate Framework)
coordinates for 2D coronal brain sections.

Features:
- Batch processing of entire folders
- Caching of predictions to JSON
- Conversion of CCF coordinates to atlas slice indices
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import json
import numpy as np

# Lazy import for DeepSlice
_deepslice_available = None


def _check_deepslice():
    """Check if DeepSlice is available."""
    global _deepslice_available
    if _deepslice_available is None:
        try:
            from DeepSlice import DSModel
            _deepslice_available = True
        except ImportError:
            _deepslice_available = False
    return _deepslice_available


class DeepSliceWrapper:
    """
    Wrapper for DeepSlice atlas position detection.

    DeepSlice predicts CCF coordinates (Oxyz, Uxyz, Vxyz) for coronal
    brain sections, enabling automatic atlas alignment.
    """

    CACHE_FILENAME = 'deepslice_positions.json'

    def __init__(self, species: str = 'mouse'):
        """
        Initialize DeepSlice wrapper.

        Args:
            species: 'mouse' or 'rat'
        """
        if not _check_deepslice():
            raise ImportError(
                "DeepSlice is not installed. "
                "Install with: pip install DeepSlice"
            )

        from DeepSlice import DSModel
        self.model = DSModel(species=species)
        self.species = species
        self._predictions = None
        self._predictions_by_filename = {}

    def predict_folder(
        self,
        folder_path: Path,
        ensemble: bool = True,
        use_cache: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Predict atlas positions for all slices in a folder.

        Args:
            folder_path: Path to folder containing brain slice images
            ensemble: Use ensemble of models (slower but more accurate)
            use_cache: Load from cache if available
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with predictions for each slice
        """
        import os
        import tempfile
        import shutil

        folder_path = Path(folder_path)

        # Get absolute path as string - don't use resolve() which can break network paths
        folder_str = str(folder_path.absolute())

        # On Windows, normalize the path
        if os.name == 'nt':
            folder_str = os.path.normpath(folder_str)

        # Verify the folder exists
        if not os.path.isdir(folder_str):
            raise ValueError(f"Folder not found or not accessible: {folder_str}")

        cache_path = folder_path / self.CACHE_FILENAME

        # Check cache
        if use_cache and cache_path.exists():
            if progress_callback:
                progress_callback("Loading cached predictions...")
            return self.load_cache(cache_path)

        # Run DeepSlice
        if progress_callback:
            progress_callback("Running DeepSlice prediction (this may take a minute)...")

        # DeepSlice has issues with network paths and some Windows paths
        # Use a try-with-fallback approach: try original path, if it fails, use temp
        temp_dir = None
        output_dir = None

        def _extract_section_number(filename: str) -> int:
            """
            Extract section number from various filename formats.

            Handles:
              - E02_01_S1.png -> 1
              - slice_s001.png -> 1
              - image_S03.tif -> 3
              - brain_section_12.nd2 -> 12
            """
            import re
            stem = Path(filename).stem

            # Pattern 1: S{N} or s{N} (e.g., E02_01_S1, slice_s001)
            match = re.search(r'[Ss](\d+)', stem)
            if match:
                return int(match.group(1))

            # Pattern 2: section_{N} or _section{N}
            match = re.search(r'section[_]?(\d+)', stem, re.IGNORECASE)
            if match:
                return int(match.group(1))

            # Pattern 3: trailing number (e.g., image_12)
            match = re.search(r'_(\d+)$', stem)
            if match:
                return int(match.group(1))

            # Pattern 4: any number at the end
            match = re.search(r'(\d+)$', stem)
            if match:
                return int(match.group(1))

            return None

        def _copy_to_temp():
            """Copy/convert images to temp folder for DeepSlice."""
            nonlocal temp_dir
            print("[DeepSlice] Preparing images for DeepSlice...")
            if progress_callback:
                progress_callback("Preparing images for DeepSlice...")
            temp_dir = Path(tempfile.mkdtemp(prefix='deepslice_'))
            print(f"[DeepSlice] Temp folder: {temp_dir}")

            # DeepSlice only supports: .jpg, .jpeg, .png
            # We need to convert other formats (ND2, TIFF) to PNG
            files_to_process = [
                f for f in sorted(folder_path.iterdir())
                if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.nd2', '.tif', '.tiff']
            ]
            total_files = len(files_to_process)
            print(f"[DeepSlice] Found {total_files} image files to process")

            # Track original filenames for mapping back later
            filename_map = {}  # deepslice_name -> original_name

            count = 0
            for i, f in enumerate(files_to_process, 1):
                suffix = f.suffix.lower()

                # Extract section number for DeepSlice-compatible naming
                section_num = _extract_section_number(f.name)
                if section_num is not None:
                    # Use DeepSlice-compatible format: slice_s{NNN}.ext
                    deepslice_name = f"slice_s{section_num:03d}"
                else:
                    # No section number found - use index
                    deepslice_name = f"slice_s{i:03d}"

                if suffix in ['.png', '.jpg', '.jpeg']:
                    # Already supported - copy with renamed filename
                    out_name = deepslice_name + suffix
                    shutil.copy2(f, temp_dir / out_name)
                    filename_map[out_name] = f.name
                    count += 1
                    print(f"[DeepSlice] ({i}/{total_files}) Copied: {f.name} -> {out_name}")

                elif suffix in ['.nd2', '.tif', '.tiff']:
                    # Need to convert to PNG
                    try:
                        print(f"[DeepSlice] ({i}/{total_files}) Converting: {f.name}...", end=" ", flush=True)

                        if suffix == '.nd2':
                            # Load ND2 file
                            try:
                                import nd2
                                with nd2.ND2File(f) as nd2_file:
                                    # Get first channel, max projection if 3D
                                    data = nd2_file.asarray()
                                    if data.ndim == 4:  # (Z, C, Y, X)
                                        img = data[:, 0, :, :].max(axis=0)
                                    elif data.ndim == 3:  # (C, Y, X) or (Z, Y, X)
                                        if data.shape[0] <= 4:  # Likely channels
                                            img = data[0]
                                        else:  # Likely Z-stack
                                            img = data.max(axis=0)
                                    else:
                                        img = data
                            except ImportError:
                                print("SKIPPED (nd2 package not installed)")
                                if progress_callback:
                                    progress_callback(f"Skipping {f.name} - nd2 package not installed")
                                continue
                        else:
                            # Load TIFF
                            import tifffile
                            data = tifffile.imread(f)
                            if data.ndim == 3:
                                img = data.max(axis=0) if data.shape[0] > 4 else data[0]
                            else:
                                img = data

                        # Normalize to 8-bit for PNG
                        img = img.astype(np.float32)
                        img_min, img_max = img.min(), img.max()
                        if img_max > img_min:
                            img = (img - img_min) / (img_max - img_min) * 255
                        img = img.astype(np.uint8)

                        # Save as PNG with DeepSlice-compatible naming
                        from skimage.io import imsave
                        out_name = deepslice_name + '.png'
                        out_path = temp_dir / out_name
                        imsave(str(out_path), img)
                        filename_map[out_name] = f.name
                        count += 1
                        print(f"OK -> {out_name}")

                    except Exception as e:
                        print(f"FAILED: {e}")
                        if progress_callback:
                            progress_callback(f"Warning: Could not convert {f.name}: {e}")
                        continue

                # Update progress callback periodically
                if progress_callback and i % 5 == 0:
                    progress_callback(f"Converting images: {i}/{total_files}")

            print(f"[DeepSlice] Prepared {count}/{total_files} images for DeepSlice")
            if progress_callback:
                progress_callback(f"Prepared {count} images for DeepSlice...")

            if count == 0:
                raise ValueError("No images could be prepared for DeepSlice")

            return str(temp_dir), filename_map

        def _run_prediction(path_to_use):
            """Run DeepSlice prediction on given path."""
            nonlocal output_dir
            print(f"[DeepSlice] Running DeepSlice neural network (ensemble={ensemble})...")
            print(f"[DeepSlice] Input path: {path_to_use}")
            print("[DeepSlice] This may take 1-5 minutes depending on number of images...")
            self.model.predict(
                path_to_use,
                ensemble=ensemble,
                section_numbers=True,  # Assume images have _sXXX naming
            )
            print("[DeepSlice] Prediction complete! Saving results...")
            # Save predictions
            if temp_dir:
                output_dir = temp_dir / 'DeepSlice_Output'
            else:
                output_dir = folder_path / 'DeepSlice_Output'
            self.model.save_predictions(str(output_dir))
            print(f"[DeepSlice] Results saved to: {output_dir}")

        # Check for obvious network paths first
        is_unc = folder_str.startswith('\\\\') or folder_str.startswith('//')

        # Check if folder contains formats that need conversion (ND2, TIFF)
        # DeepSlice only supports: .jpg, .jpeg, .png
        needs_conversion = any(
            f.suffix.lower() in ['.nd2', '.tif', '.tiff']
            for f in folder_path.iterdir()
            if f.is_file()
        )

        # On Windows, also check for mapped network drives or non-local drives
        # DeepSlice often fails with these, so we proactively use temp
        force_temp = is_unc or needs_conversion
        if os.name == 'nt' and not force_temp and len(folder_str) >= 2 and folder_str[1] == ':':
            drive_letter = folder_str[0].upper()
            # Common local drives are C, D, E. Network shares often use later letters.
            # Also check if this is a known network drive
            try:
                import subprocess
                result = subprocess.run(
                    ['net', 'use', drive_letter + ':'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    force_temp = True
                    if progress_callback:
                        progress_callback(f"Detected network drive {drive_letter}:, using temp folder...")
            except Exception:
                pass

        # filename_map for restoring original filenames after DeepSlice
        filename_map = {}

        try:
            if force_temp:
                # Known problematic path, use temp directly
                predict_path, filename_map = _copy_to_temp()
                _run_prediction(predict_path)
            else:
                # Try original path first
                try:
                    _run_prediction(folder_str)
                except (ValueError, OSError, RuntimeError) as e:
                    # If DeepSlice fails with path error, retry with temp
                    error_msg = str(e).lower()
                    if 'path' in error_msg or 'directory' in error_msg or 'not found' in error_msg or 'section number' in error_msg:
                        if progress_callback:
                            progress_callback("DeepSlice path issue detected, retrying with temp folder...")
                        predict_path, filename_map = _copy_to_temp()
                        _run_prediction(predict_path)
                    else:
                        raise  # Re-raise if it's a different error
        finally:
            # Clean up temp dir
            if temp_dir and temp_dir.exists():
                # Copy output back first
                temp_output = temp_dir / 'DeepSlice_Output'
                if temp_output.exists():
                    final_output = folder_path / 'DeepSlice_Output'
                    if final_output.exists():
                        shutil.rmtree(final_output)
                    shutil.copytree(temp_output, final_output)
                    output_dir = final_output
                shutil.rmtree(temp_dir)

        # Continue with output_dir

        # Load and process the JSON output
        json_files = list(output_dir.glob('*.json'))
        if not json_files:
            raise RuntimeError("DeepSlice did not generate output")

        # Load the first JSON file (should be the only one)
        with open(json_files[0], 'r') as f:
            raw_predictions = json.load(f)

        # Process into our format (with filename restoration)
        predictions = self._process_predictions(raw_predictions, filename_map)

        # Save to our cache
        self.save_cache(cache_path, predictions)

        if progress_callback:
            progress_callback(f"Predicted positions for {len(predictions['slices'])} slices")

        return predictions

    def _process_predictions(self, raw: Dict, filename_map: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Process raw DeepSlice output into our format.

        Args:
            raw: Raw DeepSlice output
            filename_map: Optional map of DeepSlice filenames -> original filenames
        """
        if filename_map is None:
            filename_map = {}

        slices = []

        # Handle different DeepSlice output formats
        if 'slices' in raw:
            anchors = raw['slices']
        elif 'anchors' in raw:
            anchors = raw['anchors']
        else:
            anchors = raw.get('sections', [])

        for anchor in anchors:
            # Get the filename from DeepSlice output
            ds_filename = anchor.get('filename', anchor.get('nr', 'unknown'))

            # Restore original filename if we have a map
            original_filename = filename_map.get(ds_filename, ds_filename)

            slice_info = {
                'filename': original_filename,
                'ox': anchor.get('ox', 0),
                'oy': anchor.get('oy', 0),
                'oz': anchor.get('oz', 0),  # AP position in CCF
                'ux': anchor.get('ux', 0),
                'uy': anchor.get('uy', 0),
                'uz': anchor.get('uz', 0),
                'vx': anchor.get('vx', 0),
                'vy': anchor.get('vy', 0),
                'vz': anchor.get('vz', 0),
            }

            # Calculate atlas index from oz (AP position)
            slice_info['atlas_index_10um'] = self.ccf_to_atlas_index(slice_info['oz'], 10)
            slice_info['atlas_index_25um'] = self.ccf_to_atlas_index(slice_info['oz'], 25)

            slices.append(slice_info)

            # Index by original filename
            self._predictions_by_filename[original_filename] = slice_info

        return {
            'species': self.species,
            'n_slices': len(slices),
            'slices': slices,
        }

    def ccf_to_atlas_index(
        self,
        ccf_ap_um: float,
        atlas_resolution_um: int = 10,
    ) -> int:
        """
        Convert CCF anterior-posterior position to atlas slice index.

        The CCF coordinate system has:
        - Origin at the anterior tip
        - Z axis running anterior-posterior
        - Units in micrometers

        Args:
            ccf_ap_um: AP position in micrometers (oz coordinate)
            atlas_resolution_um: Atlas resolution (10, 25, or 50 um)

        Returns:
            Atlas slice index (0-based)
        """
        # CCF z ranges from ~0 (anterior) to ~13200um (posterior) for mouse
        # Atlas index = position / resolution
        index = int(ccf_ap_um / atlas_resolution_um)
        return max(0, index)

    def get_position_for_slice(
        self,
        filename: str,
        atlas_resolution_um: int = 10,
    ) -> Optional[int]:
        """
        Get atlas position for a specific slice by filename.

        Args:
            filename: Image filename
            atlas_resolution_um: Atlas resolution

        Returns:
            Atlas slice index, or None if not found
        """
        # Try exact match first
        if filename in self._predictions_by_filename:
            info = self._predictions_by_filename[filename]
            return self.ccf_to_atlas_index(info['oz'], atlas_resolution_um)

        # Try matching by basename
        basename = Path(filename).name
        for key, info in self._predictions_by_filename.items():
            if Path(key).name == basename:
                return self.ccf_to_atlas_index(info['oz'], atlas_resolution_um)

        return None

    def get_slice_info(self, filename: str) -> Optional[Dict]:
        """Get full prediction info for a slice."""
        if filename in self._predictions_by_filename:
            return self._predictions_by_filename[filename]

        basename = Path(filename).name
        for key, info in self._predictions_by_filename.items():
            if Path(key).name == basename:
                return info

        return None

    def get_all_positions(
        self,
        atlas_resolution_um: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get atlas positions for all predicted slices.

        Returns:
            List of dicts with filename and atlas_index
        """
        result = []
        for info in self._predictions_by_filename.values():
            result.append({
                'filename': info['filename'],
                'atlas_index': self.ccf_to_atlas_index(info['oz'], atlas_resolution_um),
                'ccf_ap_um': info['oz'],
            })
        return result

    def save_cache(self, cache_path: Path, predictions: Dict):
        """Save predictions to cache file."""
        with open(cache_path, 'w') as f:
            json.dump(predictions, f, indent=2)

    def load_cache(self, cache_path: Path) -> Dict[str, Any]:
        """Load predictions from cache file."""
        with open(cache_path, 'r') as f:
            predictions = json.load(f)

        # Rebuild filename index
        self._predictions_by_filename = {}
        for slice_info in predictions.get('slices', []):
            self._predictions_by_filename[slice_info['filename']] = slice_info

        return predictions


def is_deepslice_available() -> bool:
    """Check if DeepSlice is installed and available."""
    return _check_deepslice()


def predict_single_slice_position(
    image: np.ndarray,
    temp_dir: Optional[Path] = None,
) -> Optional[Dict]:
    """
    Predict position for a single slice.

    Note: DeepSlice works better with batches of ~30+ slices.
    Single-slice prediction may be less accurate.

    Args:
        image: 2D image array
        temp_dir: Temporary directory to save image

    Returns:
        Position info dict or None
    """
    import tempfile
    from skimage.io import imsave

    if not _check_deepslice():
        return None

    # Create temp directory if not provided
    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp())

    # Save image to temp file
    temp_path = temp_dir / 'slice_s001.tif'
    imsave(str(temp_path), image)

    # Run prediction
    wrapper = DeepSliceWrapper()
    try:
        predictions = wrapper.predict_folder(temp_dir, ensemble=False, use_cache=False)
        if predictions['slices']:
            return predictions['slices'][0]
    except Exception as e:
        print(f"[DeepSlice] Single slice prediction failed: {e}")

    return None
