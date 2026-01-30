"""
Pure matplotlib visualization module for colocalization QC.

No Qt dependencies. Each function returns a matplotlib Figure that can be
displayed in a Qt widget or saved to disk.
"""

import matplotlib
matplotlib.use('Agg')  # Headless rendering

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from skimage.segmentation import find_boundaries
from pathlib import Path
import pandas as pd


def create_overlay_image(green_channel, nuclei_labels, measurements_df, figsize=(12, 10)):
    """
    Create overlay showing green channel with colored nucleus boundaries.

    Parameters
    ----------
    green_channel : ndarray (Y, X)
        Signal intensity image (uint8 or uint16)
    nuclei_labels : ndarray (Y, X)
        Integer label image from StarDist (0=background)
    measurements_df : DataFrame
        Must contain columns: label, is_positive
    figsize : tuple
        Figure size in inches

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Normalize green channel for display
    green_norm = green_channel.astype(float)
    green_norm = (green_norm - green_norm.min()) / (green_norm.max() - green_norm.min() + 1e-8)

    # Create RGB overlay image
    rgb_overlay = np.stack([green_norm, green_norm, green_norm], axis=-1)

    # Find boundaries
    boundaries = find_boundaries(nuclei_labels, mode='outer')

    # Separate positive and negative labels
    positive_labels = set(measurements_df[measurements_df['is_positive']]['label'].values)
    negative_labels = set(measurements_df[~measurements_df['is_positive']]['label'].values)

    # Create label-to-color mapping
    boundary_mask = boundaries & (nuclei_labels > 0)
    boundary_label_ids = nuclei_labels[boundary_mask]

    # Color boundaries
    for label_id in np.unique(boundary_label_ids):
        if label_id == 0:
            continue

        label_boundary_mask = boundary_mask & (nuclei_labels == label_id)

        if label_id in positive_labels:
            # Lime green for positive
            rgb_overlay[label_boundary_mask] = [0, 1.0, 0]  # #00FF00
        elif label_id in negative_labels:
            # Red for negative
            rgb_overlay[label_boundary_mask] = [1.0, 0, 0]  # #FF0000

    # Display
    ax.imshow(rgb_overlay)
    ax.axis('off')

    n_pos = len(positive_labels)
    n_neg = len(negative_labels)
    ax.set_title(f"Colocalization Overlay — {n_pos} positive (lime), {n_neg} negative (red)",
                 fontsize=14, pad=10)

    fig.tight_layout()
    return fig


def create_background_mask_overlay(green_channel, nuclei_labels, tissue_mask, figsize=(10, 8)):
    """
    Show tissue mask and excluded nuclei regions for background estimation.

    Parameters
    ----------
    green_channel : ndarray (Y, X)
        Signal intensity image
    nuclei_labels : ndarray (Y, X)
        Integer label image
    tissue_mask : ndarray (Y, X), bool
        True where tissue is estimated
    figsize : tuple
        Figure size in inches

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Normalize green channel
    green_norm = green_channel.astype(float)
    green_norm = (green_norm - green_norm.min()) / (green_norm.max() - green_norm.min() + 1e-8)

    # Create RGB overlay
    rgb_overlay = np.stack([green_norm, green_norm, green_norm], axis=-1)

    # Overlay tissue mask as semi-transparent blue
    tissue_mask_bool = tissue_mask.astype(bool)
    rgb_overlay[tissue_mask_bool] = rgb_overlay[tissue_mask_bool] * 0.8 + np.array([0, 0, 1.0]) * 0.2

    # Overlay nuclei as semi-transparent red
    nuclei_mask = nuclei_labels > 0
    rgb_overlay[nuclei_mask] = rgb_overlay[nuclei_mask] * 0.7 + np.array([1.0, 0, 0]) * 0.3

    ax.imshow(rgb_overlay)
    ax.axis('off')
    ax.set_title("Background Estimation Mask — blue=tissue, red=excluded nuclei",
                 fontsize=14, pad=10)

    fig.tight_layout()
    return fig


def create_fold_change_histogram(measurements_df, threshold, background, figsize=(8, 5)):
    """
    Histogram of fold change values with threshold line.

    Parameters
    ----------
    measurements_df : DataFrame
        Must contain columns: fold_change, is_positive
    threshold : float
        Fold-change threshold (e.g., 2.0)
    background : float
        Estimated background intensity
    figsize : tuple
        Figure size in inches

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)

    fold_changes = measurements_df['fold_change'].values
    is_positive = measurements_df['is_positive'].values

    # Create histogram bins
    n_bins = 50
    hist_range = (fold_changes.min(), fold_changes.max())

    # Plot histogram with colored bars
    counts, bins, patches = ax.hist(fold_changes, bins=n_bins, range=hist_range,
                                     edgecolor='black', linewidth=0.5)

    # Color bars based on threshold
    bin_centers = (bins[:-1] + bins[1:]) / 2
    for patch, bin_center in zip(patches, bin_centers):
        if bin_center >= threshold:
            patch.set_facecolor('green')
        else:
            patch.set_facecolor('red')

    # Add threshold line
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold}x)')

    # Add fold_change=1.0 line
    ax.axvline(1.0, color='gray', linestyle=':', linewidth=1.5, label='Fold Change = 1.0')

    # Calculate statistics
    n_positive = is_positive.sum()
    n_negative = (~is_positive).sum()
    total = len(is_positive)
    pct_positive = 100 * n_positive / total if total > 0 else 0
    pct_negative = 100 * n_negative / total if total > 0 else 0

    # Add text annotation
    text_str = f"Positive: {n_positive} ({pct_positive:.1f}%)\nNegative: {n_negative} ({pct_negative:.1f}%)"
    ax.text(0.98, 0.98, text_str, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10)

    ax.set_xlabel("Fold Change over Background", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Fold Change Distribution (bg={background:.1f}, threshold={threshold}x)",
                 fontsize=12, pad=10)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def create_intensity_scatter(measurements_df, background, threshold, figsize=(8, 6)):
    """
    Scatter plot of nucleus area vs mean intensity.

    Parameters
    ----------
    measurements_df : DataFrame
        Must contain columns: area, mean_intensity, is_positive
    background : float
        Background intensity value
    threshold : float
        Fold-change threshold
    figsize : tuple
        Figure size in inches

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Separate positive and negative
    positive_df = measurements_df[measurements_df['is_positive']]
    negative_df = measurements_df[~measurements_df['is_positive']]

    # Plot positive cells
    if len(positive_df) > 0:
        ax.scatter(positive_df['area'], positive_df['mean_intensity'],
                  c='lime', alpha=0.6, s=20, label=f'Positive (n={len(positive_df)})')

    # Plot negative cells
    if len(negative_df) > 0:
        ax.scatter(negative_df['area'], negative_df['mean_intensity'],
                  c='red', alpha=0.6, s=20, label=f'Negative (n={len(negative_df)})')

    # Add background line
    ax.axhline(background, color='blue', linestyle='--', linewidth=1.5,
               label=f'Background ({background:.1f})')

    # Add threshold line
    threshold_intensity = background * threshold
    ax.axhline(threshold_intensity, color='black', linestyle='--', linewidth=1.5,
               label=f'Threshold ({threshold_intensity:.1f})')

    ax.set_xlabel("Nucleus Area (pixels)", fontsize=11)
    ax.set_ylabel("Mean Signal Intensity", fontsize=11)
    ax.set_title("Intensity vs Size", fontsize=12, pad=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def create_roi_summary_bar(roi_counts_list, figsize=(8, 5)):
    """
    Grouped bar chart of positive/negative counts per ROI.

    Parameters
    ----------
    roi_counts_list : list of dict
        Each dict has keys: roi, total, positive, negative, fraction
        Skip rows where roi == "TOTAL"
    figsize : tuple
        Figure size in inches

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Filter out TOTAL row
    roi_data = [r for r in roi_counts_list if r['roi'] != 'TOTAL']

    if len(roi_data) == 0:
        ax.text(0.5, 0.5, 'No ROI data available',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        return fig

    # Extract data
    roi_names = [r['roi'] for r in roi_data]
    positive_counts = [r['positive'] for r in roi_data]
    negative_counts = [r['negative'] for r in roi_data]
    fractions = [r['fraction'] for r in roi_data]

    # Bar positions
    x = np.arange(len(roi_names))
    width = 0.35

    # Create bars
    bars_pos = ax.bar(x - width/2, positive_counts, width, label='Positive', color='green', alpha=0.8)
    bars_neg = ax.bar(x + width/2, negative_counts, width, label='Negative', color='red', alpha=0.8)

    # Add fraction labels on top of positive bars
    for i, (bar, frac) in enumerate(zip(bars_pos, fractions)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{frac:.1%}', ha='center', va='bottom', fontsize=8)

    # Formatting
    ax.set_xlabel('ROI', fontsize=11)
    ax.set_ylabel('Cell Count', fontsize=11)
    ax.set_title('ROI Cell Counts', fontsize=12, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(roi_names, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    return fig


def save_all_qc_figures(output_dir, green_channel, nuclei_labels, measurements_df,
                        tissue_mask, threshold, background, roi_counts=None,
                        prefix="qc", dpi=150):
    """
    Generate and save all QC figures to disk.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save figures
    green_channel : ndarray
        Signal intensity image
    nuclei_labels : ndarray
        Label image
    measurements_df : DataFrame
        Measurements table
    tissue_mask : ndarray
        Tissue mask
    threshold : float
        Fold-change threshold
    background : float
        Background intensity
    roi_counts : list of dict, optional
        ROI summary data
    prefix : str
        Prefix for saved filenames
    dpi : int
        Resolution for saved images

    Returns
    -------
    list of Path
        Paths to saved figure files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    # 1. Overlay image
    fig = create_overlay_image(green_channel, nuclei_labels, measurements_df)
    path = output_dir / f"{prefix}_overlay.png"
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    saved_paths.append(path)

    # 2. Background mask
    fig = create_background_mask_overlay(green_channel, nuclei_labels, tissue_mask)
    path = output_dir / f"{prefix}_background_mask.png"
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    saved_paths.append(path)

    # 3. Fold change histogram
    fig = create_fold_change_histogram(measurements_df, threshold, background)
    path = output_dir / f"{prefix}_fold_change_histogram.png"
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    saved_paths.append(path)

    # 4. Intensity scatter
    fig = create_intensity_scatter(measurements_df, background, threshold)
    path = output_dir / f"{prefix}_intensity_scatter.png"
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    saved_paths.append(path)

    # 5. ROI summary (if provided)
    if roi_counts is not None:
        fig = create_roi_summary_bar(roi_counts)
        path = output_dir / f"{prefix}_roi_summary.png"
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        saved_paths.append(path)

    return saved_paths
