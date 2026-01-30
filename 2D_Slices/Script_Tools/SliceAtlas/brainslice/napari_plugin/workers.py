"""
workers.py - Background worker threads for BrainSlice napari plugin

Provides QThread workers for long-running operations:
- Image loading
- Nuclei detection
- Colocalization analysis
"""

from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from qtpy.QtCore import QThread, Signal


class ImageLoaderWorker(QThread):
    """Load ND2/TIFF images in background thread."""

    progress = Signal(str)
    finished = Signal(bool, str, object, object, object)  # success, msg, red, green, metadata

    def __init__(
        self,
        file_path: Path,
        red_idx: int = 0,
        green_idx: int = 1,
        z_projection: str = 'max',
    ):
        super().__init__()
        self.file_path = Path(file_path)
        self.red_idx = red_idx
        self.green_idx = green_idx
        self.z_projection = z_projection

    def run(self):
        try:
            from ..core.io import load_image, extract_channels

            self.progress.emit(f"Loading {self.file_path.name}...")
            data, metadata = load_image(self.file_path, z_projection=self.z_projection)

            self.progress.emit("Extracting channels...")
            red, green = extract_channels(data, self.red_idx, self.green_idx)

            self.finished.emit(True, "Load complete", red, green, metadata)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e), None, None, None)


class FolderLoaderWorker(QThread):
    """Load folder of images as a stack in background thread."""

    progress = Signal(int, int, str)  # current, total, filename
    finished = Signal(bool, str, object, object)  # success, msg, stack, metadata

    def __init__(
        self,
        folder_path: Path,
        red_idx: int = 0,
        green_idx: int = 1,
        z_projection: str = 'max',
    ):
        super().__init__()
        self.folder_path = Path(folder_path)
        self.red_idx = red_idx
        self.green_idx = green_idx
        self.z_projection = z_projection

    def run(self):
        try:
            print(f"[FolderLoaderWorker] Starting to load folder: {self.folder_path}")
            from ..core.io import load_folder

            def progress_callback(current, total, filename):
                print(f"[FolderLoaderWorker] Loading {current}/{total}: {filename}")
                self.progress.emit(current, total, filename)

            stack, metadata = load_folder(
                self.folder_path,
                z_projection=self.z_projection,
                progress_callback=progress_callback,
            )

            print(f"[FolderLoaderWorker] Successfully loaded stack with shape: {stack.shape}")
            self.finished.emit(True, "Folder loaded as stack", stack, metadata)

        except Exception as e:
            import traceback
            print(f"[FolderLoaderWorker] ERROR: {e}")
            traceback.print_exc()
            self.finished.emit(False, str(e), None, None)


class DetectionWorker(QThread):
    """Run StarDist nuclei detection in background thread."""

    progress = Signal(str)
    finished = Signal(bool, str, int, object)  # success, msg, count, labels

    def __init__(
        self,
        image: np.ndarray,
        params: Dict[str, Any],
    ):
        super().__init__()
        self.image = image
        self.params = params

    def run(self):
        try:
            from ..core.detection import NucleiDetector

            model_name = self.params.get('model', '2D_versatile_fluo')
            self.progress.emit(f"Loading StarDist model: {model_name}...")

            detector = NucleiDetector(model_name=model_name)

            self.progress.emit("Detecting nuclei...")
            labels, details = detector.detect(
                self.image,
                prob_thresh=self.params.get('prob_thresh', 0.5),
                nms_thresh=self.params.get('nms_thresh', 0.4),
                scale=self.params.get('scale', 1.0),
            )

            # Filter by size if requested
            if self.params.get('filter_size', True):
                self.progress.emit("Filtering by size...")
                labels = detector.filter_by_size(
                    labels,
                    min_area=self.params.get('min_area', 50),
                    max_area=self.params.get('max_area', 5000),
                )

            count = len(np.unique(labels)) - 1  # Exclude background
            self.progress.emit(f"Detected {count} nuclei")

            self.finished.emit(True, f"Detected {count} nuclei", count, labels)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e), 0, None)


class ColocalizationWorker(QThread):
    """Run colocalization analysis in background thread."""

    progress = Signal(str)
    finished = Signal(bool, str, object, object, object)  # success, msg, measurements_df, summary, tissue_mask

    def __init__(
        self,
        signal_image: np.ndarray,
        nuclei_labels: np.ndarray,
        params: Dict[str, Any],
    ):
        super().__init__()
        self.signal_image = signal_image
        self.nuclei_labels = nuclei_labels
        self.params = params

    def run(self):
        try:
            from ..core.colocalization import ColocalizationAnalyzer

            print(f"[ColocWorker] Starting colocalization...")
            print(f"[ColocWorker]   signal_image: shape={self.signal_image.shape}, dtype={self.signal_image.dtype}, range=[{self.signal_image.min()}, {self.signal_image.max()}]")
            print(f"[ColocWorker]   nuclei_labels: shape={self.nuclei_labels.shape}, max_label={self.nuclei_labels.max()}")
            print(f"[ColocWorker]   params: {self.params}")

            bg_method = self.params.get('background_method', 'percentile')
            bg_percentile = self.params.get('background_percentile', 10.0)

            self.progress.emit(f"Estimating background ({bg_method})...")

            analyzer = ColocalizationAnalyzer(
                background_method=bg_method,
                background_percentile=bg_percentile,
            )

            dilation = self.params.get('dilation_iterations', 50)
            background = analyzer.estimate_background(
                self.signal_image,
                self.nuclei_labels,
                dilation_iterations=dilation,
            )

            tissue_mask = analyzer.estimate_tissue_mask(self.nuclei_labels, dilation)

            self.progress.emit("Measuring intensities...")
            measurements = analyzer.measure_nuclei_intensities(
                self.signal_image,
                self.nuclei_labels,
            )

            self.progress.emit("Classifying positive/negative...")
            classified = analyzer.classify_positive_negative(
                measurements,
                background,
                method=self.params.get('threshold_method', 'fold_change'),
                threshold=self.params.get('threshold_value', 2.0),
            )

            summary = analyzer.get_summary_statistics(classified)

            msg = (f"Analysis complete: {summary['positive_cells']} positive, "
                   f"{summary['negative_cells']} negative "
                   f"({summary['positive_fraction']*100:.1f}% positive)")

            self.finished.emit(True, msg, classified, summary, tissue_mask)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e), None, None, None)


class QuantificationWorker(QThread):
    """Run regional quantification in background thread."""

    progress = Signal(str)
    finished = Signal(bool, str, object, object, object)  # success, msg, cell_df, region_df, summary

    def __init__(
        self,
        cell_measurements,  # DataFrame
        atlas_labels: np.ndarray,
        atlas_manager: Optional[Any] = None,
        output_dir: Optional[Path] = None,
        sample_id: str = 'sample',
    ):
        super().__init__()
        self.cell_measurements = cell_measurements
        self.atlas_labels = atlas_labels
        self.atlas_manager = atlas_manager
        self.output_dir = output_dir
        self.sample_id = sample_id

    def run(self):
        try:
            from ..core.quantification import RegionQuantifier

            self.progress.emit("Assigning cells to regions...")

            quantifier = RegionQuantifier(self.atlas_manager)

            cell_data = quantifier.assign_cells_to_regions(
                self.cell_measurements,
                self.atlas_labels,
            )

            self.progress.emit("Counting per region...")
            region_counts = quantifier.count_per_region(cell_data)

            summary = quantifier.get_summary(cell_data, region_counts)

            # Export if output directory provided
            if self.output_dir:
                self.progress.emit("Exporting results...")
                quantifier.export_results(
                    cell_data, region_counts,
                    self.output_dir, self.sample_id
                )

            msg = (f"Quantification complete: {summary['total_cells']} cells "
                   f"in {summary['regions_with_cells']} regions")

            self.finished.emit(True, msg, cell_data, region_counts, summary)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e), None, None, None)
