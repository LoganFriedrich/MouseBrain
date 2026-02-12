"""
evaluate_pipeline.py -- Layered smoke-test for the detection/colocalization pipeline.

Each layer validates ONE component. If a layer fails, the script tells you
exactly what broke and skips everything that depends on it.

Usage:
    conda activate "Y:\2_Connectome\envs\braintool"
    cd Y:\2_Connectome\Tissue\2D_Slices\Script_Tools\SliceAtlas
    python evaluate_pipeline.py                         # synthetic data (no image needed)
    python evaluate_pipeline.py path/to/your/image.nd2  # real image
"""

import sys
import time
import traceback
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

_layer_results = {}   # name → (pass/fail/skip, detail)

def run_layer(name, depends_on=None):
    """Decorator: run a test layer, skip if dependencies failed."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            # Check dependencies
            if depends_on:
                deps = depends_on if isinstance(depends_on, list) else [depends_on]
                for dep in deps:
                    if dep not in _layer_results or _layer_results[dep][0] != 'PASS':
                        msg = f"skipped (depends on '{dep}' which failed)"
                        _layer_results[name] = ('SKIP', msg)
                        print(f"\n{'='*60}")
                        print(f"LAYER: {name}")
                        print(f"  SKIP — {msg}")
                        return None

            print(f"\n{'='*60}")
            print(f"LAYER: {name}")
            print(f"{'='*60}")
            t0 = time.time()
            try:
                result = fn(*args, **kwargs)
                elapsed = time.time() - t0
                _layer_results[name] = ('PASS', f'{elapsed:.1f}s')
                print(f"  PASS ({elapsed:.1f}s)")
                return result
            except Exception as e:
                elapsed = time.time() - t0
                _layer_results[name] = ('FAIL', str(e))
                print(f"  FAIL ({elapsed:.1f}s)")
                print(f"  Error: {e}")
                traceback.print_exc()
                return None
        wrapper.__name__ = name
        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────
# LAYER 0: Imports
# ─────────────────────────────────────────────────────────────

@run_layer("0_imports")
def layer_0_imports():
    """Can we import the core modules at all?"""
    results = {}

    # Core detection
    from brainslice.core.detection import (
        NucleiDetector,
        preprocess_for_detection,
        PRETRAINED_MODELS,
    )
    results['NucleiDetector'] = True
    results['preprocess_for_detection'] = True
    results['PRETRAINED_MODELS'] = list(PRETRAINED_MODELS.keys())
    print(f"  detection.py OK — models: {results['PRETRAINED_MODELS']}")

    # Cellpose backend
    try:
        from brainslice.core.cellpose_backend import CellposeDetector
        results['CellposeDetector'] = True
        print(f"  cellpose_backend.py OK — CellposeDetector importable")
    except ImportError as e:
        results['CellposeDetector'] = False
        print(f"  cellpose_backend.py — import OK but cellpose not installed: {e}")

    # Colocalization
    from brainslice.core.colocalization import ColocalizationAnalyzer
    results['ColocalizationAnalyzer'] = True
    print(f"  colocalization.py OK — ColocalizationAnalyzer importable")

    # Visualization
    from brainslice.core.visualization import (
        create_overlay_image,
        create_background_surface_plot,
        create_fold_change_histogram,
        create_gmm_diagnostic,
    )
    results['visualization'] = True
    print(f"  visualization.py OK — all plot functions importable")

    # IO
    from brainslice.core.io import load_image, extract_channels
    results['io'] = True
    print(f"  io.py OK")

    return results


# ─────────────────────────────────────────────────────────────
# LAYER 1: Preprocessing on synthetic data
# ─────────────────────────────────────────────────────────────

@run_layer("1_preprocessing", depends_on="0_imports")
def layer_1_preprocessing():
    """Does preprocessing run without error on a synthetic image?"""
    from brainslice.core.detection import preprocess_for_detection

    # Create a synthetic image: gradient background + random bright spots
    rng = np.random.default_rng(42)
    h, w = 512, 512
    # Gradient background simulates uneven illumination
    bg = np.linspace(100, 300, w).reshape(1, w) * np.ones((h, 1))
    # Scattered bright nuclei
    nuclei = np.zeros((h, w))
    for _ in range(50):
        cy, cx = rng.integers(20, h-20), rng.integers(20, w-20)
        yy, xx = np.ogrid[-15:16, -15:16]
        mask = (yy**2 + xx**2) <= 10**2
        nuclei[cy-15:cy+16, cx-15:cx+16][mask] += rng.uniform(200, 500)
    image = (bg + nuclei + rng.normal(0, 10, (h, w))).clip(0, 65535).astype(np.uint16)

    # Test each preprocessing step individually
    configs = [
        dict(background_subtraction=True, bg_sigma=50.0),
        dict(clahe=True, clahe_clip_limit=0.02),
        dict(gaussian_sigma=1.5),
        dict(background_subtraction=True, clahe=True, gaussian_sigma=1.0),
    ]

    for i, cfg in enumerate(configs):
        result = preprocess_for_detection(image, **cfg)
        assert result.dtype == np.float32, f"Config {i}: expected float32, got {result.dtype}"
        assert result.shape == image.shape, f"Config {i}: shape mismatch"
        assert result.min() >= 0, f"Config {i}: negative values"
        assert result.max() <= 1.0 + 1e-6, f"Config {i}: values > 1"
        label = ', '.join(f'{k}={v}' for k, v in cfg.items())
        print(f"  config [{label}]: shape={result.shape}, range=[{result.min():.3f}, {result.max():.3f}]")

    return image  # pass synthetic image to next layers


# ─────────────────────────────────────────────────────────────
# LAYER 2: StarDist detection on synthetic data
# ─────────────────────────────────────────────────────────────

@run_layer("2_stardist_detection", depends_on="1_preprocessing")
def layer_2_stardist_detection(image):
    """Can StarDist load a model and detect nuclei?"""
    from brainslice.core.detection import NucleiDetector, preprocess_for_detection

    # Preprocess first
    preprocessed = preprocess_for_detection(image, background_subtraction=True, clahe=True)

    detector = NucleiDetector(model_name='2D_versatile_fluo')
    print(f"  Model loaded: {detector.model_name}")

    labels, details = detector.detect(preprocessed, prob_thresh=0.3, nms_thresh=0.4)

    n_detected = labels.max()
    print(f"  Raw detections: {n_detected}")
    print(f"  Labels shape: {labels.shape}, dtype: {labels.dtype}")
    print(f"  Details keys: {list(details.keys())}")
    if 'prob' in details:
        probs = details['prob']
        print(f"  Probability range: [{min(probs):.3f}, {max(probs):.3f}] (n={len(probs)})")

    assert labels.shape == image.shape
    assert n_detected >= 0, "Detection returned negative label count"

    return detector, labels, details, preprocessed


# ─────────────────────────────────────────────────────────────
# LAYER 3: Post-detection filters
# ─────────────────────────────────────────────────────────────

@run_layer("3_post_filters", depends_on="2_stardist_detection")
def layer_3_post_filters(detector, labels, details, preprocessed):
    """Do all post-detection filters work?"""
    n_raw = labels.max()
    print(f"  Starting with {n_raw} detections")

    # Size filter
    sized = detector.filter_by_size(labels, min_area=30, max_area=5000)
    n_after_size = sized.max()
    print(f"  After size filter (30-5000): {n_after_size} ({n_raw - n_after_size} removed)")

    # Circularity filter
    circular = detector.filter_by_circularity(sized, min_circularity=0.3)
    n_after_circ = circular.max()
    print(f"  After circularity filter (>0.3): {n_after_circ} ({n_after_size - n_after_circ} removed)")

    # Confidence filter
    conf_filtered, n_conf_removed = detector.filter_by_confidence(labels, details, min_confidence=0.5)
    print(f"  Confidence filter (>0.5): removed {n_conf_removed}")

    # Border-touching filter
    border_filtered, n_border_removed = detector.filter_border_touching(labels)
    print(f"  Border filter: removed {n_border_removed}")

    # Morphology filter
    morph_filtered, n_morph_removed = detector.filter_by_morphology(
        labels, intensity_image=preprocessed, min_solidity=0.7
    )
    print(f"  Morphology filter (solidity>0.7): removed {n_morph_removed}")

    # Auto n_tiles
    from brainslice.core.detection import NucleiDetector
    tiles_small = NucleiDetector.compute_n_tiles((512, 512), tile_size=1024)
    tiles_large = NucleiDetector.compute_n_tiles((4096, 4096), tile_size=1024)
    print(f"  Auto n_tiles: 512x512 → {tiles_small}, 4096x4096 → {tiles_large}")

    return sized  # return the size-filtered result for coloc testing


# ─────────────────────────────────────────────────────────────
# LAYER 4: Cellpose detection (optional)
# ─────────────────────────────────────────────────────────────

@run_layer("4_cellpose_detection", depends_on="1_preprocessing")
def layer_4_cellpose_detection(image):
    """Can Cellpose load and detect? (optional — not blocking)"""
    from brainslice.core.cellpose_backend import CellposeDetector
    from brainslice.core.detection import preprocess_for_detection

    preprocessed = preprocess_for_detection(image, background_subtraction=True, clahe=True)

    detector = CellposeDetector(model_name='nuclei', gpu=True)
    print(f"  Cellpose model loaded: nuclei")

    labels, details = detector.detect(preprocessed, diameter=20.0)

    n_detected = labels.max()
    print(f"  Cellpose detections: {n_detected}")
    print(f"  Labels shape: {labels.shape}, dtype: {labels.dtype}")
    print(f"  Details keys: {list(details.keys())}")

    return labels


# ─────────────────────────────────────────────────────────────
# LAYER 5: Colocalization — global background
# ─────────────────────────────────────────────────────────────

@run_layer("5_coloc_global", depends_on="3_post_filters")
def layer_5_coloc_global(labels, original_image):
    """Does global background estimation + fold_change classification work?"""
    from brainslice.core.colocalization import ColocalizationAnalyzer

    # Create synthetic signal: ~30% of nuclei are "positive"
    rng = np.random.default_rng(99)
    h, w = original_image.shape
    signal = rng.normal(50, 10, (h, w)).clip(0).astype(np.float32)  # low background
    # Make some nuclei bright in signal channel
    positive_labels = set()
    for lbl in range(1, labels.max() + 1):
        if rng.random() < 0.3:  # 30% positive
            signal[labels == lbl] += rng.uniform(100, 300)
            positive_labels.add(lbl)
    print(f"  Synthetic signal: {len(positive_labels)} / {labels.max()} nuclei made bright")

    analyzer = ColocalizationAnalyzer(background_method='percentile', background_percentile=10)

    # Global background
    bg = analyzer.estimate_background(signal, labels, dilation_iterations=20)
    print(f"  Global background estimate: {bg:.2f}")

    # Tissue mask
    tissue_mask = analyzer.estimate_tissue_mask(labels, dilation_iterations=20)
    print(f"  Tissue mask: {tissue_mask.sum()} tissue pixels / {h*w} total")

    # Measure intensities
    measurements = analyzer.measure_nuclei_intensities(signal, labels)
    print(f"  Measurements: {len(measurements)} nuclei measured")
    print(f"  Columns: {list(measurements.columns)}")

    # Classify — fold_change
    classified = analyzer.classify_positive_negative(
        measurements, bg, method='fold_change', threshold=2.0
    )
    n_pos = classified['is_positive'].sum()
    n_total = len(classified)
    print(f"  fold_change classification: {n_pos}/{n_total} positive ({100*n_pos/n_total:.1f}%)")
    assert 'is_positive' in classified.columns
    assert 'fold_change' in classified.columns
    assert 'background' in classified.columns

    # Summary
    summary = analyzer.get_summary_statistics(classified, bg)
    print(f"  Summary: {summary}")

    return analyzer, signal, labels, classified, bg, tissue_mask


# ─────────────────────────────────────────────────────────────
# LAYER 6: Colocalization — local background
# ─────────────────────────────────────────────────────────────

@run_layer("6_coloc_local_bg", depends_on="5_coloc_global")
def layer_6_coloc_local_bg(analyzer, signal, labels):
    """Does local background estimation produce a 2D surface and classify correctly?"""

    # Local background
    bg_surface = analyzer.estimate_local_background(
        signal, labels, block_size=128, min_tissue_pixels=20
    )
    print(f"  Local background surface: shape={bg_surface.shape}, dtype={bg_surface.dtype}")
    print(f"  Background range: [{bg_surface.min():.2f}, {bg_surface.max():.2f}], mean={bg_surface.mean():.2f}")
    assert bg_surface.shape == signal.shape
    assert bg_surface.ndim == 2

    # Classify with 2D background
    measurements = analyzer.measure_nuclei_intensities(signal, labels)
    classified = analyzer.classify_positive_negative(
        measurements, bg_surface, method='fold_change', threshold=2.0
    )
    n_pos = classified['is_positive'].sum()
    n_total = len(classified)
    print(f"  Local bg classification: {n_pos}/{n_total} positive ({100*n_pos/n_total:.1f}%)")

    # Check that per-nucleus background varies (not all same value)
    bg_vals = classified['background'].values
    if len(bg_vals) > 1:
        bg_range = bg_vals.max() - bg_vals.min()
        print(f"  Per-nucleus bg range: {bg_range:.2f} (should be > 0 if illumination varies)")

    return bg_surface


# ─────────────────────────────────────────────────────────────
# LAYER 7: Colocalization — area_fraction method
# ─────────────────────────────────────────────────────────────

@run_layer("7_coloc_area_fraction", depends_on="5_coloc_global")
def layer_7_coloc_area_fraction(analyzer, signal, labels, bg):
    """Does area_fraction classification work?"""

    measurements = analyzer.measure_nuclei_intensities(signal, labels)
    classified = analyzer.classify_positive_negative(
        measurements, bg,
        method='area_fraction', threshold=2.0,
        signal_image=signal, nuclei_labels=labels,
        area_fraction=0.5,
    )
    n_pos = classified['is_positive'].sum()
    n_total = len(classified)
    print(f"  area_fraction classification: {n_pos}/{n_total} positive ({100*n_pos/n_total:.1f}%)")
    assert 'positive_pixel_fraction' in classified.columns
    print(f"  Pixel fraction range: [{classified['positive_pixel_fraction'].min():.3f}, "
          f"{classified['positive_pixel_fraction'].max():.3f}]")

    return classified


# ─────────────────────────────────────────────────────────────
# LAYER 8: Visualization plots
# ─────────────────────────────────────────────────────────────

@run_layer("8_visualization", depends_on=["5_coloc_global", "6_coloc_local_bg"])
def layer_8_visualization(signal, labels, classified, bg, tissue_mask, bg_surface):
    """Do all diagnostic plots render without error?"""
    import matplotlib
    matplotlib.use('Agg')

    from brainslice.core.visualization import (
        create_overlay_image,
        create_background_mask_overlay,
        create_background_surface_plot,
        create_fold_change_histogram,
        create_intensity_scatter,
    )
    import matplotlib.pyplot as plt

    plots_ok = []

    # Fold change histogram
    fig = create_fold_change_histogram(classified, threshold=2.0, background=float(bg))
    plt.close(fig)
    plots_ok.append('fold_change_histogram')

    # Intensity scatter
    fig = create_intensity_scatter(classified, float(bg), 2.0)
    plt.close(fig)
    plots_ok.append('intensity_scatter')

    # Overlay image
    fig = create_overlay_image(signal, labels, classified)
    plt.close(fig)
    plots_ok.append('overlay_image')

    # Background mask
    fig = create_background_mask_overlay(signal, labels, tissue_mask)
    plt.close(fig)
    plots_ok.append('background_mask')

    # Background surface (NEW)
    fig = create_background_surface_plot(bg_surface, nuclei_labels=labels)
    plt.close(fig)
    plots_ok.append('background_surface')

    print(f"  All {len(plots_ok)} plots rendered OK: {plots_ok}")
    return plots_ok


# ─────────────────────────────────────────────────────────────
# LAYER 9: Real image (optional)
# ─────────────────────────────────────────────────────────────

@run_layer("9_real_image", depends_on="0_imports")
def layer_9_real_image(image_path):
    """Load a real image and run the full pipeline on it."""
    from brainslice.core.io import load_image, extract_channels
    from brainslice.core.detection import NucleiDetector, preprocess_for_detection
    from brainslice.core.colocalization import ColocalizationAnalyzer

    print(f"  Loading: {image_path}")
    data, metadata = load_image(str(image_path))
    print(f"  Shape: {data.shape}, dtype: {data.dtype}")
    print(f"  Metadata: {metadata.get('channels', 'unknown channels')}")

    # Extract channels (assume red=0, green=1 — adjust if needed)
    if data.ndim == 3 and data.shape[0] >= 2:
        nuclear = data[0]
        signal = data[1]
    elif data.ndim == 2:
        nuclear = data
        signal = data
        print("  WARNING: single-channel image, using same for nuclear and signal")
    else:
        print(f"  ERROR: unexpected shape {data.shape}, cannot extract channels")
        raise ValueError(f"Unexpected image shape: {data.shape}")

    print(f"  Nuclear channel: shape={nuclear.shape}, range=[{nuclear.min()}, {nuclear.max()}]")
    print(f"  Signal channel: shape={signal.shape}, range=[{signal.min()}, {signal.max()}]")

    # Preprocess
    preprocessed = preprocess_for_detection(
        nuclear, background_subtraction=True, clahe=True, gaussian_sigma=1.0
    )
    print(f"  Preprocessed: range=[{preprocessed.min():.3f}, {preprocessed.max():.3f}]")

    # Detect
    detector = NucleiDetector(model_name='2D_versatile_fluo')
    n_tiles = NucleiDetector.compute_n_tiles(preprocessed.shape)
    print(f"  Auto n_tiles: {n_tiles}")

    labels, details = detector.detect(preprocessed, prob_thresh=0.4, nms_thresh=0.4, n_tiles=n_tiles)
    n_raw = labels.max()
    print(f"  Raw detections: {n_raw}")

    # Filter
    labels = detector.filter_by_size(labels, min_area=50, max_area=5000)
    labels, n_border = detector.filter_border_touching(labels)
    labels, n_morph = detector.filter_by_morphology(labels, intensity_image=nuclear, min_solidity=0.7)
    n_final = labels.max()
    print(f"  After filters: {n_final} (removed border={n_border}, morph={n_morph})")

    # Colocalize
    analyzer = ColocalizationAnalyzer(background_method='percentile')
    bg = analyzer.estimate_background(signal.astype(np.float32), labels, dilation_iterations=30)
    print(f"  Global background: {bg:.2f}")

    measurements = analyzer.measure_nuclei_intensities(signal.astype(np.float32), labels)
    classified = analyzer.classify_positive_negative(measurements, bg, threshold=2.0)
    n_pos = classified['is_positive'].sum()
    print(f"  Classification: {n_pos}/{n_final} positive ({100*n_pos/n_final:.1f}%)")

    # Local background
    bg_surface = analyzer.estimate_local_background(signal.astype(np.float32), labels, block_size=256)
    bg_range = bg_surface.max() - bg_surface.min()
    print(f"  Local background variation: {bg_range:.2f}")

    classified_local = analyzer.classify_positive_negative(
        measurements, bg_surface, threshold=2.0
    )
    n_pos_local = classified_local['is_positive'].sum()
    print(f"  Local bg classification: {n_pos_local}/{n_final} positive ({100*n_pos_local/n_final:.1f}%)")
    print(f"  Difference: {abs(n_pos - n_pos_local)} cells changed classification")

    return {
        'n_raw': n_raw, 'n_final': n_final,
        'n_positive_global': int(n_pos), 'n_positive_local': int(n_pos_local),
        'bg_global': float(bg), 'bg_local_range': float(bg_range),
    }


# ─────────────────────────────────────────────────────────────
# LAYER 10: napari widget import (no display)
# ─────────────────────────────────────────────────────────────

@run_layer("10_widget_import", depends_on="0_imports")
def layer_10_widget_import():
    """Can the napari widget class be imported without crashing?"""
    # This imports Qt, so it may fail in headless environments
    try:
        from brainslice.napari_plugin.main_widget import BrainSliceWidget
        print(f"  BrainSliceWidget importable")
        return True
    except ImportError as e:
        if 'qt' in str(e).lower() or 'napari' in str(e).lower():
            print(f"  Expected in headless: {e}")
            print(f"  Widget import requires a display — this is OK for console testing")
            return False
        raise


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    print("BrainSlice Pipeline Evaluation")
    print("=" * 60)
    if image_path:
        print(f"Real image: {image_path}")
    else:
        print("No image provided — running on synthetic data only.")
        print("  To test with a real image: python evaluate_pipeline.py path/to/image.nd2")
    print()

    # Run layers in dependency order
    layer_0_imports()

    synthetic_image = layer_1_preprocessing()

    stardist_result = layer_2_stardist_detection(synthetic_image)
    if stardist_result:
        detector, labels, details, preprocessed = stardist_result
        filtered_labels = layer_3_post_filters(detector, labels, details, preprocessed)
    else:
        filtered_labels = None

    # Cellpose is independent — doesn't block anything
    layer_4_cellpose_detection(synthetic_image)

    coloc_result = None
    if filtered_labels is not None:
        coloc_result = layer_5_coloc_global(filtered_labels, synthetic_image)

    local_bg_surface = None
    if coloc_result:
        analyzer, signal, labels, classified, bg, tissue_mask = coloc_result
        local_bg_surface = layer_6_coloc_local_bg(analyzer, signal, labels)
        layer_7_coloc_area_fraction(analyzer, signal, labels, bg)

        if local_bg_surface is not None:
            layer_8_visualization(signal, labels, classified, bg, tissue_mask, local_bg_surface)

    # Real image (if provided)
    if image_path and image_path.exists():
        layer_9_real_image(image_path)
    elif image_path:
        print(f"\n  WARNING: Image not found: {image_path}")

    # Widget import (independent)
    layer_10_widget_import()

    # ─────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────
    print("\n")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for s, _ in _layer_results.values() if s == 'PASS')
    failed = sum(1 for s, _ in _layer_results.values() if s == 'FAIL')
    skipped = sum(1 for s, _ in _layer_results.values() if s == 'SKIP')

    for name, (status, detail) in _layer_results.items():
        icon = {'PASS': '+', 'FAIL': 'X', 'SKIP': '-'}[status]
        print(f"  [{icon}] {name}: {status} ({detail})")

    print(f"\n  {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\n  FIX THE FIRST FAILURE — downstream layers depend on it.")
        for name, (status, detail) in _layer_results.items():
            if status == 'FAIL':
                print(f"  >> First failure: {name}")
                print(f"     Error: {detail}")
                break

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
