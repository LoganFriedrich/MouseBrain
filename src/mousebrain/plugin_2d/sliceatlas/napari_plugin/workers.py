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
    """Run nuclei detection with preprocessing and filtering in background thread."""

    progress = Signal(str)
    # success, msg, count, labels, metrics_dict
    finished = Signal(bool, str, int, object, object)

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
            metrics = {
                'raw_count': 0,
                'filtered_count': 0,
                'removed_by_size': 0,
                'removed_by_border': 0,
                'removed_by_confidence': 0,
                'removed_by_morphology': 0,
                'backend': 'threshold',
                'preprocessing': {},
            }

            image = self.image
            backend = self.params.get('backend', 'threshold')
            metrics['backend'] = backend

            # ══════════════════════════════════════════════════════════
            # THRESHOLD BACKEND (default — for sparse fluorescent nuclei)
            # ══════════════════════════════════════════════════════════
            if backend == 'threshold':
                from ..core.detection import detect_by_threshold

                self.progress.emit("Detecting nuclei (threshold)...")
                labels, details = detect_by_threshold(
                    image,
                    method=self.params.get('threshold_method', 'otsu'),
                    percentile=self.params.get('threshold_percentile', 99.0),
                    manual_threshold=self.params.get('manual_threshold', None),
                    min_area=self.params.get('min_area', 10),
                    max_area=self.params.get('max_area', 5000),
                    opening_radius=self.params.get('opening_radius', 0),
                    closing_radius=self.params.get('closing_radius', 0),
                    fill_holes=self.params.get('fill_holes', True),
                    split_touching=self.params.get('split_touching', False),
                    split_footprint_size=self.params.get('split_footprint_size', 10),
                    gaussian_sigma=self.params.get('gaussian_sigma', 1.0),
                    use_hysteresis=self.params.get('use_hysteresis', True),
                    hysteresis_low_fraction=self.params.get('hysteresis_low_fraction', 0.5),
                    min_solidity=self.params.get('min_solidity', 0.0),
                    min_circularity=self.params.get('min_circularity', 0.0),
                )

                raw_count = details.get('raw_count', 0)
                final_count = details.get('filtered_count', 0)
                metrics['raw_count'] = raw_count
                metrics['filtered_count'] = final_count
                metrics['removed_by_size'] = details.get('removed_by_size', 0)
                metrics['removed_by_morphology'] = details.get('removed_by_morphology', 0)
                metrics['n_watershed_splits'] = details.get('n_watershed_splits', 0)
                metrics['threshold_value'] = details.get('threshold', 0)
                metrics['threshold_low'] = details.get('threshold_low', 0)
                metrics['threshold_method'] = details.get('method', 'otsu')
                metrics['use_hysteresis'] = details.get('use_hysteresis', False)

            # ══════════════════════════════════════════════════════════
            # STARDIST / CELLPOSE BACKENDS (for dense nuclei fields)
            # ══════════════════════════════════════════════════════════
            else:
                from ..core.detection import NucleiDetector, preprocess_for_detection

                # Preprocessing
                preproc_params = {}
                bg_sub = self.params.get('background_subtraction', False)
                use_clahe = self.params.get('clahe', False)
                gauss_sigma = self.params.get('gaussian_sigma', 0.0)

                if bg_sub or use_clahe or gauss_sigma > 0:
                    self.progress.emit("Preprocessing image...")
                    image = preprocess_for_detection(
                        image,
                        background_subtraction=bg_sub,
                        bg_sigma=self.params.get('bg_sigma', 50.0),
                        clahe=use_clahe,
                        clahe_clip_limit=self.params.get('clahe_clip_limit', 0.02),
                        clahe_kernel_size=self.params.get('clahe_kernel_size', 128),
                        gaussian_sigma=gauss_sigma,
                    )
                    preproc_params = {
                        'background_subtraction': bg_sub,
                        'bg_sigma': self.params.get('bg_sigma', 50.0),
                        'clahe': use_clahe,
                        'clahe_clip_limit': self.params.get('clahe_clip_limit', 0.02),
                        'gaussian_sigma': gauss_sigma,
                    }
                metrics['preprocessing'] = preproc_params

                model_name = self.params.get('model', '2D_versatile_fluo')

                if backend == 'cellpose':
                    self.progress.emit(f"Loading Cellpose model: {model_name}...")
                    try:
                        from ..core.cellpose_backend import CellposeDetector
                        cp_detector = CellposeDetector(model_name=model_name)
                        self.progress.emit("Detecting nuclei (Cellpose)...")
                        labels, details = cp_detector.detect(
                            image,
                            diameter=self.params.get('diameter', 30),
                        )
                    except ImportError:
                        self.progress.emit("Cellpose not available, falling back to StarDist...")
                        backend = 'stardist'
                        metrics['backend'] = 'stardist (cellpose unavailable)'

                if backend == 'stardist':
                    self.progress.emit(f"Loading StarDist model: {model_name}...")
                    detector = NucleiDetector(model_name=model_name)

                    n_tiles = self.params.get('n_tiles', None)
                    if n_tiles is None and self.params.get('auto_n_tiles', True):
                        n_tiles = NucleiDetector.compute_n_tiles(image.shape)

                    self.progress.emit("Detecting nuclei (StarDist)...")
                    labels, details = detector.detect(
                        image,
                        prob_thresh=self.params.get('prob_thresh', 0.5),
                        nms_thresh=self.params.get('nms_thresh', 0.4),
                        scale=self.params.get('scale', 1.0),
                        n_tiles=n_tiles,
                    )

                raw_count = len(np.unique(labels)) - 1
                metrics['raw_count'] = raw_count
                self.progress.emit(f"Raw detections: {raw_count}")

                # Post-detection filtering (StarDist/Cellpose only — threshold
                # backend does its own size filtering internally)

                # Size filter
                if self.params.get('filter_size', True):
                    self.progress.emit("Filtering by size...")
                    count_before = len(np.unique(labels)) - 1
                    from skimage.measure import regionprops
                    min_area = self.params.get('min_area', 50)
                    max_area = self.params.get('max_area', 5000)
                    props = regionprops(labels)
                    valid_labels = [p.label for p in props if min_area <= p.area <= max_area]
                    filtered = np.zeros_like(labels)
                    for new_id, old_label in enumerate(valid_labels, 1):
                        filtered[labels == old_label] = new_id
                    labels = filtered
                    count_after = len(np.unique(labels)) - 1
                    metrics['removed_by_size'] = count_before - count_after

                # Border filter
                if self.params.get('remove_border', False):
                    self.progress.emit("Removing border-touching nuclei...")
                    from skimage.segmentation import clear_border
                    count_before = len(np.unique(labels)) - 1
                    cleared = clear_border(labels, buffer_size=self.params.get('border_width', 1))
                    unique_ids = np.unique(cleared)
                    unique_ids = unique_ids[unique_ids > 0]
                    relabeled = np.zeros_like(cleared)
                    for new_id, old_label in enumerate(unique_ids, 1):
                        relabeled[cleared == old_label] = new_id
                    labels = relabeled
                    count_after = len(np.unique(labels)) - 1
                    metrics['removed_by_border'] = count_before - count_after

                # Morphology filter
                min_solidity = self.params.get('min_solidity', 0.0)
                if min_solidity > 0:
                    self.progress.emit("Filtering by morphology...")
                    count_before = len(np.unique(labels)) - 1
                    from skimage.measure import regionprops as rp
                    props = rp(labels)
                    valid = [p.label for p in props if p.solidity >= min_solidity]
                    filtered = np.zeros_like(labels)
                    for new_id, old_label in enumerate(valid, 1):
                        filtered[labels == old_label] = new_id
                    labels = filtered
                    count_after = len(np.unique(labels)) - 1
                    metrics['removed_by_morphology'] = count_before - count_after

            # ── Final count and size stats (all backends) ──
            final_count = len(np.unique(labels)) - 1
            if 'filtered_count' not in metrics or backend != 'threshold':
                metrics['filtered_count'] = final_count

            if final_count > 0:
                from skimage.measure import regionprops as rp_final
                areas = [p.area for p in rp_final(labels)]
                metrics['size_stats'] = {
                    'mean': float(np.mean(areas)),
                    'median': float(np.median(areas)),
                    'std': float(np.std(areas)),
                    'min': float(np.min(areas)),
                    'max': float(np.max(areas)),
                }

            self.progress.emit(f"Detected {final_count} nuclei")
            self.finished.emit(
                True,
                f"Detected {final_count} nuclei",
                final_count,
                labels,
                metrics,
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e), 0, None, None)


class ColocalizationWorker(QThread):
    """Run colocalization analysis in background thread."""

    progress = Signal(str)
    finished = Signal(bool, str, object, object, object)  # success, msg, measurements_df, summary, tissue_mask

    def __init__(
        self,
        signal_image: np.ndarray,
        nuclei_labels: np.ndarray,
        params: Dict[str, Any],
        signal_image_for_area: Optional[np.ndarray] = None,
        labels_for_area: Optional[np.ndarray] = None,
        nuclear_image: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.signal_image = signal_image
        self.nuclei_labels = nuclei_labels
        self.params = params
        self.signal_image_for_area = signal_image_for_area
        self.labels_for_area = labels_for_area
        self.nuclear_image = nuclear_image

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

            # Use local or global background estimation
            if self.params.get('use_local_background', False):
                self.progress.emit("Estimating local background...")
                background = analyzer.estimate_local_background(
                    self.signal_image,
                    self.nuclei_labels,
                    dilation_iterations=dilation,
                    block_size=self.params.get('bg_block_size', 256),
                )
                # Store for background surface visualization
                self.background_surface = background
            else:
                background = analyzer.estimate_background(
                    self.signal_image,
                    self.nuclei_labels,
                    dilation_iterations=dilation,
                )
                self.background_surface = None

            tissue_mask = analyzer.estimate_tissue_mask(self.nuclei_labels, dilation)

            soma_dilation = self.params.get('soma_dilation', 0)
            if soma_dilation > 0:
                self.progress.emit(f"Measuring soma intensities (dilation={soma_dilation}px)...")
                measurements = analyzer.measure_soma_intensities(
                    self.signal_image,
                    self.nuclei_labels,
                    soma_dilation=soma_dilation,
                )
            else:
                self.progress.emit("Measuring nuclear intensities...")
                measurements = analyzer.measure_nuclei_intensities(
                    self.signal_image,
                    self.nuclei_labels,
                )

            self.progress.emit("Classifying positive/negative...")
            thresh_method = self.params.get('threshold_method', 'fold_change')
            classify_kwargs = {
                'method': thresh_method,
                'threshold': self.params.get('threshold_value', 2.0),
            }

            # For background_mean method, pass sigma_threshold, signal_image, labels
            if thresh_method == 'background_mean':
                classify_kwargs['sigma_threshold'] = self.params.get('sigma_threshold', 0)
                classify_kwargs['signal_image'] = self.signal_image
                classify_kwargs['nuclei_labels'] = self.nuclei_labels

            # For area_fraction method, pass the signal image and labels
            if thresh_method == 'area_fraction':
                classify_kwargs['signal_image'] = self.signal_image_for_area
                classify_kwargs['nuclei_labels'] = self.labels_for_area
                classify_kwargs['area_fraction'] = self.params.get('area_fraction', 0.5)

            classified = analyzer.classify_positive_negative(
                measurements,
                background,
                **classify_kwargs,
            )

            summary = analyzer.get_summary_statistics(classified)

            # Compute Manders/Pearson validation metrics
            if self.nuclear_image is not None:
                self.progress.emit("Computing colocalization metrics (Pearson, Manders)...")
                from ..core.colocalization import compute_colocalization_metrics
                coloc_metrics = compute_colocalization_metrics(
                    red_image=self.nuclear_image,
                    green_image=self.signal_image,
                    nuclei_labels=self.nuclei_labels,
                    background_green=summary['background_used'],
                    tissue_mask=tissue_mask,
                    soma_dilation=soma_dilation,
                )
                summary['coloc_metrics'] = coloc_metrics
                print(f"[ColocWorker] Colocalization metrics: {coloc_metrics}")

            # Store diagnostics for the widget to pick up after signal
            self.background_diagnostics = analyzer.background_diagnostics
            # Store tissue pixels used for background estimation (for GMM plot)
            nuclei_mask = self.nuclei_labels > 0
            tissue_outside = tissue_mask & (~nuclei_mask)
            self.tissue_pixels = self.signal_image[tissue_outside] if tissue_outside.any() else self.signal_image[tissue_mask]

            msg = (f"Analysis complete: {summary['positive_cells']} positive, "
                   f"{summary['negative_cells']} negative "
                   f"({summary['positive_fraction']*100:.1f}% positive)")

            self.finished.emit(True, msg, classified, summary, tissue_mask)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, str(e), None, None, None)


class DualColocalizationWorker(QThread):
    """Run dual-channel colocalization analysis in background thread."""

    progress = Signal(str)
    finished = Signal(bool, str, object, object, object)  # success, msg, measurements_df, summary, tissue_mask

    def __init__(
        self,
        signal_image_1: np.ndarray,   # red / mCherry
        signal_image_2: np.ndarray,   # green / eYFP
        nuclei_labels: np.ndarray,
        params_ch1: Dict[str, Any],
        params_ch2: Dict[str, Any],
    ):
        super().__init__()
        self.signal_image_1 = signal_image_1
        self.signal_image_2 = signal_image_2
        self.nuclei_labels = nuclei_labels
        self.params_ch1 = params_ch1
        self.params_ch2 = params_ch2

    def run(self):
        try:
            from ..core.colocalization import analyze_dual_colocalization

            print(f"[DualColocWorker] Starting dual-channel colocalization...")
            print(f"[DualColocWorker]   ch1 (red): shape={self.signal_image_1.shape}, range=[{self.signal_image_1.min()}, {self.signal_image_1.max()}]")
            print(f"[DualColocWorker]   ch2 (green): shape={self.signal_image_2.shape}, range=[{self.signal_image_2.min()}, {self.signal_image_2.max()}]")
            print(f"[DualColocWorker]   labels: max_label={self.nuclei_labels.max()}")
            print(f"[DualColocWorker]   params_ch1: {self.params_ch1}")
            print(f"[DualColocWorker]   params_ch2: {self.params_ch2}")

            self.progress.emit("Analyzing channel 1 (red / mCherry)...")

            classified, summary = analyze_dual_colocalization(
                signal_image_1=self.signal_image_1,
                signal_image_2=self.signal_image_2,
                nuclei_labels=self.nuclei_labels,
                # Ch1 params
                background_method_1=self.params_ch1.get('background_method', 'gmm'),
                background_percentile_1=self.params_ch1.get('background_percentile', 10.0),
                threshold_method_1=self.params_ch1.get('threshold_method', 'fold_change'),
                threshold_value_1=self.params_ch1.get('threshold_value', 2.0),
                cell_body_dilation_1=self.params_ch1.get('dilation_iterations', 10),
                area_fraction_1=self.params_ch1.get('area_fraction', 0.5),
                soma_dilation_1=self.params_ch1.get('soma_dilation', 6),
                sigma_threshold_1=self.params_ch1.get('sigma_threshold', 0),
                # Ch2 params
                background_method_2=self.params_ch2.get('background_method', 'gmm'),
                background_percentile_2=self.params_ch2.get('background_percentile', 10.0),
                threshold_method_2=self.params_ch2.get('threshold_method', 'fold_change'),
                threshold_value_2=self.params_ch2.get('threshold_value', 2.0),
                cell_body_dilation_2=self.params_ch2.get('dilation_iterations', 50),
                area_fraction_2=self.params_ch2.get('area_fraction', 0.5),
                soma_dilation_2=self.params_ch2.get('soma_dilation', 5),
                ch1_name='red',
                ch2_name='green',
            )

            self.progress.emit("Dual-channel analysis complete.")

            # Store diagnostics
            self.background_diagnostics_ch1 = summary.get('bg_diagnostics_red', {})
            self.background_diagnostics_ch2 = summary.get('bg_diagnostics_green', {})

            n_dual = summary.get('n_dual', 0)
            n_r = summary.get('n_red_only', 0)
            n_g = summary.get('n_green_only', 0)
            total = summary.get('total_nuclei', 0)

            msg = (f"Dual analysis complete: {n_dual} dual+, "
                   f"{n_r} red-only, {n_g} green-only "
                   f"({total} total)")

            self.finished.emit(True, msg, classified, summary, None)

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
