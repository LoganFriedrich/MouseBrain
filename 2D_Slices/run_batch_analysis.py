#!/usr/bin/env python3
"""
Standalone batch 2D slice analysis — no mousebrain package required.

Processes all .nd2 files under ENCR/, runs StarDist nuclei detection,
measures eYFP colocalization with GMM background estimation,
and saves labeled overlay images + CSV measurements.

Usage:
    python run_batch_analysis.py
    python run_batch_analysis.py --folder "Y:/2_Connectome/Tissue/2D_Slices/ENCR/ENCR_02_01_ATLAS"
    python run_batch_analysis.py --file "Y:/2_Connectome/Tissue/2D_Slices/ENCR/ENCR_02_01_ATLAS/E02_01_S1.nd2"
"""

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# Fix broken TF __version__ (corrupted install)
try:
    import tensorflow as _tf
    if not hasattr(_tf, '__version__'):
        import importlib.metadata
        _tf.__version__ = importlib.metadata.version('tensorflow')
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # No GUI needed
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from skimage.measure import regionprops_table, label as sk_label
from scipy.ndimage import binary_dilation

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

ROOT = Path(r"Y:\2_Connectome\Tissue\2D_Slices\ENCR")
NUCLEAR_CH = 0     # Red / H2B-mCherry
SIGNAL_CH = 1      # Green / eYFP
STARDIST_MODEL = "2D_versatile_fluo"
PROB_THRESH = 0.5
NMS_THRESH = 0.4
MIN_AREA = 50
MAX_AREA = 5000
FOLD_CHANGE_THRESH = 2.0
DILATION_ITERATIONS = 20
SCALE = 1.0

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_nd2(path):
    """Load nd2 file, return (nuclear_channel, signal_channel)."""
    import nd2
    data = nd2.imread(str(path))
    # nd2 files can be (C, Y, X) or (Y, X, C) or (Z, C, Y, X)
    if data.ndim == 3:
        if data.shape[0] <= 4:  # (C, Y, X)
            nuclear = data[NUCLEAR_CH]
            signal = data[SIGNAL_CH]
        else:  # (Y, X, C)
            nuclear = data[:, :, NUCLEAR_CH]
            signal = data[:, :, SIGNAL_CH]
    elif data.ndim == 4:
        # Take max projection over Z if 4D: (Z, C, Y, X)
        nuclear = np.max(data[:, NUCLEAR_CH], axis=0)
        signal = np.max(data[:, SIGNAL_CH], axis=0)
    elif data.ndim == 2:
        # Single channel - use as both
        nuclear = data
        signal = data
    else:
        raise ValueError(f"Unexpected shape: {data.shape}")

    return nuclear.astype(np.float32), signal.astype(np.float32)


def detect_nuclei(nuclear_img):
    """Detect nuclei using scikit-image watershed (no TF required).

    Falls back from StarDist to watershed if TF is unavailable or hangs.
    Watershed + distance transform handles most non-overlapping nuclei well.
    """
    from skimage.filters import threshold_otsu, gaussian
    from skimage.morphology import remove_small_objects, remove_small_holes, disk, opening
    from skimage.segmentation import watershed
    from scipy.ndimage import distance_transform_edt, label as ndi_label
    from skimage.feature import peak_local_max

    # Percentile normalize
    lo, hi = np.percentile(nuclear_img, [1, 99.8])
    if hi <= lo:
        hi = lo + 1
    img_norm = np.clip((nuclear_img - lo) / (hi - lo), 0, 1)

    # Smooth to reduce noise
    img_smooth = gaussian(img_norm, sigma=2)

    # Threshold
    try:
        thresh = threshold_otsu(img_smooth)
    except ValueError:
        thresh = 0.3
    binary = img_smooth > thresh

    # Clean up
    binary = opening(binary, disk(2))
    binary = remove_small_objects(binary, min_size=MIN_AREA)
    binary = remove_small_holes(binary, area_threshold=200)

    # Distance transform + watershed to split touching nuclei
    distance = distance_transform_edt(binary)
    # Find peaks (nuclei centers) — min_distance controls splitting
    coords = peak_local_max(distance, min_distance=8, labels=binary)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi_label(mask)
    labels = watershed(-distance, markers, mask=binary)

    # Filter by size
    props = regionprops_table(labels, properties=['label', 'area'])
    keep = set()
    for lbl, area in zip(props['label'], props['area']):
        if MIN_AREA <= area <= MAX_AREA:
            keep.add(lbl)

    filtered = np.where(np.isin(labels, list(keep)), labels, 0)
    return filtered


def estimate_background_gmm(signal_img, labels):
    """Estimate background using GMM on tissue pixels outside nuclei."""
    nuclei_mask = labels > 0
    tissue_mask = binary_dilation(nuclei_mask, iterations=DILATION_ITERATIONS)
    tissue_outside = tissue_mask & (~nuclei_mask)

    if tissue_outside.sum() < 100:
        tissue_outside = tissue_mask

    pixels = signal_img[tissue_outside].ravel()

    if len(pixels) > 200000:
        rng = np.random.RandomState(42)
        pixels = rng.choice(pixels, 200000, replace=False)

    try:
        from sklearn.mixture import GaussianMixture

        X = pixels.reshape(-1, 1)
        gmm1 = GaussianMixture(n_components=1, random_state=42).fit(X)
        gmm2 = GaussianMixture(n_components=2, random_state=42).fit(X)

        bic1 = gmm1.bic(X)
        bic2 = gmm2.bic(X)

        if bic2 < bic1:
            means = gmm2.means_.flatten()
            stds = np.sqrt(gmm2.covariances_.flatten())
            bg_idx = np.argmin(means)
            bg_mean = means[bg_idx]
            bg_std = stds[bg_idx]
            method_used = "gmm_2component"
            separation = abs(means[1] - means[0]) / stds[bg_idx] if stds[bg_idx] > 0 else 0
        else:
            bg_mean = gmm1.means_.flatten()[0]
            bg_std = np.sqrt(gmm1.covariances_.flatten()[0])
            method_used = "gmm_1component"
            separation = None
    except Exception:
        bg_mean = np.percentile(pixels, 10)
        bg_std = np.std(pixels)
        method_used = "percentile_fallback"
        separation = None

    return bg_mean, bg_std, method_used, separation, pixels


def measure_and_classify(signal_img, labels, background):
    """Measure signal in each nucleus and classify pos/neg."""
    props = regionprops_table(labels, intensity_image=signal_img,
                              properties=['label', 'centroid', 'area', 'mean_intensity', 'max_intensity'])

    df = pd.DataFrame(props)
    df.rename(columns={'centroid-0': 'y', 'centroid-1': 'x'}, inplace=True)

    # Compute median per nucleus (not in regionprops)
    medians = []
    for lbl in df['label']:
        mask = labels == lbl
        medians.append(np.median(signal_img[mask]))
    df['median_intensity'] = medians

    df['fold_change'] = df['mean_intensity'] / background if background > 0 else 0
    df['is_positive'] = df['fold_change'] >= FOLD_CHANGE_THRESH
    df['background_value'] = background

    return df


def create_overlay(signal_img, labels, df, sample_name):
    """Create overlay image with colored cell boundaries."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Show signal channel
    vmin, vmax = np.percentile(signal_img, [1, 99.5])
    ax.imshow(signal_img, cmap='gray', vmin=vmin, vmax=vmax)

    # Draw boundaries
    from skimage.segmentation import find_boundaries
    boundaries = find_boundaries(labels, mode='outer')

    pos_labels = set(df[df['is_positive']]['label'])

    # Vectorized: one boundary pass, then color by membership
    all_boundaries = find_boundaries(labels, mode='outer')
    overlay = np.zeros((*labels.shape, 4), dtype=np.float32)
    # Get label at each boundary pixel (dilate labels by 1 to cover boundary)
    from scipy.ndimage import maximum_filter
    label_at_boundary = maximum_filter(labels, size=3)
    pos_mask = np.isin(label_at_boundary, list(pos_labels)) & all_boundaries
    neg_mask = all_boundaries & ~pos_mask
    overlay[pos_mask] = [0, 1, 0, 0.8]   # Lime green
    overlay[neg_mask] = [1, 0, 0, 0.5]   # Red, dimmer

    ax.imshow(overlay)

    n_pos = df['is_positive'].sum()
    n_total = len(df)
    frac = n_pos / n_total * 100 if n_total > 0 else 0
    ax.set_title(f"{sample_name}\n{n_pos} positive / {n_total} total ({frac:.1f}%)", fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    return fig


def create_annotated_overlay(signal_img, labels, df, sample_name, top_n=30):
    """Overlay with arrows pointing to brightest positive cells."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))

    vmin, vmax = np.percentile(signal_img, [1, 99.5])
    ax.imshow(signal_img, cmap='gray', vmin=vmin, vmax=vmax)

    from skimage.segmentation import find_boundaries
    from scipy.ndimage import maximum_filter
    pos_labels = set(df[df['is_positive']]['label'])
    all_boundaries = find_boundaries(labels, mode='outer')
    overlay = np.zeros((*labels.shape, 4), dtype=np.float32)
    label_at_boundary = maximum_filter(labels, size=3)
    pos_mask = np.isin(label_at_boundary, list(pos_labels)) & all_boundaries
    neg_mask = all_boundaries & ~pos_mask
    overlay[pos_mask] = [0, 1, 0, 0.8]
    overlay[neg_mask] = [1, 0, 0, 0.4]
    ax.imshow(overlay)

    # Arrow annotations for top positive cells
    top_pos = df[df['is_positive']].nlargest(top_n, 'fold_change')
    for _, row in top_pos.iterrows():
        ax.annotate(
            f"{row['fold_change']:.1f}x",
            xy=(row['x'], row['y']),
            xytext=(row['x'] + 40, row['y'] - 40),
            fontsize=6, color='yellow', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='yellow', lw=1),
        )

    n_pos = df['is_positive'].sum()
    n_total = len(df)
    frac = n_pos / n_total * 100 if n_total > 0 else 0
    ax.set_title(f"{sample_name} — Top {min(top_n, len(top_pos))} Positive Cells\n"
                 f"{n_pos}/{n_total} positive ({frac:.1f}%)", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    return fig


def create_fold_change_histogram(df, background, sample_name):
    """Histogram of fold-change values."""
    fig, ax = plt.subplots(figsize=(8, 5))

    fc = df['fold_change'].values
    fc_pos = fc[df['is_positive']]
    fc_neg = fc[~df['is_positive']]

    bins = np.linspace(0, min(fc.max(), 20), 60)
    ax.hist(fc_neg, bins=bins, color='#cc4444', alpha=0.7, label=f'Negative ({len(fc_neg)})')
    ax.hist(fc_pos, bins=bins, color='#44cc44', alpha=0.7, label=f'Positive ({len(fc_pos)})')
    ax.axvline(FOLD_CHANGE_THRESH, color='white', linestyle='--', linewidth=2, label=f'Threshold ({FOLD_CHANGE_THRESH}x)')

    ax.set_xlabel('Fold Change (vs. tissue background)')
    ax.set_ylabel('Number of Cells')
    ax.set_title(f'{sample_name}\nBackground = {background:.1f}')
    ax.legend()
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

    plt.tight_layout()
    return fig


def process_one_image(nd2_path, output_dir):
    """Full pipeline for one image. Returns (success, summary_dict)."""
    sample_name = nd2_path.stem
    sample_output = output_dir / sample_name
    sample_output.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"  Loading {nd2_path.name}...", end=" ", flush=True)

    try:
        nuclear, signal = load_nd2(nd2_path)
    except Exception as e:
        print(f"LOAD FAILED: {e}")
        return False, {"sample": sample_name, "error": str(e)}

    print(f"({nuclear.shape}) ", end="", flush=True)

    # Detect
    print("detecting...", end=" ", flush=True)
    labels = detect_nuclei(nuclear)
    n_nuclei = len(np.unique(labels)) - 1  # exclude 0
    print(f"({n_nuclei} nuclei) ", end="", flush=True)

    if n_nuclei == 0:
        print("NO NUCLEI FOUND")
        return False, {"sample": sample_name, "error": "no nuclei"}

    # Background
    print("background...", end=" ", flush=True)
    bg_mean, bg_std, bg_method, separation, tissue_pixels = estimate_background_gmm(signal, labels)
    print(f"({bg_method}, bg={bg_mean:.1f}) ", end="", flush=True)

    # Measure & classify
    df = measure_and_classify(signal, labels, bg_mean)
    n_pos = df['is_positive'].sum()
    n_neg = len(df) - n_pos
    frac = n_pos / len(df) * 100 if len(df) > 0 else 0
    print(f"classify...({n_pos}+/{n_neg}-  {frac:.1f}%) ", end="", flush=True)

    # Save CSV
    df.to_csv(sample_output / f"{sample_name}_measurements.csv", index=False)

    # Save figures
    print("saving...", end=" ", flush=True)

    fig1 = create_overlay(signal, labels, df, sample_name)
    fig1.savefig(sample_output / f"{sample_name}_overlay.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)

    fig2 = create_annotated_overlay(signal, labels, df, sample_name)
    fig2.savefig(sample_output / f"{sample_name}_annotated.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)

    fig3 = create_fold_change_histogram(df, bg_mean, sample_name)
    fig3.savefig(sample_output / f"{sample_name}_histogram.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)

    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s)")

    return True, {
        "sample": sample_name,
        "nuclei": n_nuclei,
        "positive": n_pos,
        "negative": n_neg,
        "fraction_positive": round(frac, 2),
        "background": round(bg_mean, 2),
        "bg_method": bg_method,
        "duration_s": round(elapsed, 1),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Batch 2D slice colocalization analysis")
    parser.add_argument("--folder", type=Path, help="Process all .nd2 in this folder")
    parser.add_argument("--file", type=Path, help="Process a single .nd2 file")
    parser.add_argument("--root", type=Path, default=ROOT, help="Root folder to scan for .nd2 files")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    # Find files
    if args.file:
        nd2_files = [args.file]
    elif args.folder:
        nd2_files = sorted(args.folder.glob("*.nd2"))
    else:
        nd2_files = sorted(args.root.rglob("*.nd2"))

    if not nd2_files:
        print("No .nd2 files found!")
        return 1

    output_dir = args.output or (args.root / "batch_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"Batch 2D Slice Analysis")
    print(f"  Files: {len(nd2_files)}")
    print(f"  Output: {output_dir}")
    print(f"  Detection: watershed (scikit-image)")
    print(f"  Threshold: {FOLD_CHANGE_THRESH}x fold change")
    print(f"{'='*70}\n")

    results = []
    for i, f in enumerate(nd2_files):
        print(f"[{i+1}/{len(nd2_files)}] ", end="")
        success, summary = process_one_image(f, output_dir)
        results.append(summary)

    # Save summary
    summary_df = pd.DataFrame(results)
    summary_path = output_dir / "batch_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Print summary
    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE")
    print(f"{'='*70}")
    if 'nuclei' in summary_df.columns:
        valid = summary_df.dropna(subset=['nuclei'])
        print(f"  Processed: {len(valid)}/{len(nd2_files)}")
        print(f"  Total nuclei: {int(valid['nuclei'].sum()):,}")
        print(f"  Total positive: {int(valid['positive'].sum()):,}")
        print(f"  Mean fraction: {valid['fraction_positive'].mean():.1f}%")
    print(f"\n  Summary CSV: {summary_path}")
    print(f"  Results dir: {output_dir}")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
