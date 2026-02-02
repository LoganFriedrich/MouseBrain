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


def create_annotated_overlay(green_channel, nuclei_labels, measurements_df,
                             max_annotations=30, figsize=(14, 12)):
    """
    Create overlay with arrow annotations pointing to colocalized (positive) cells.

    Shows the green channel with all nuclei boundaries colored (lime=positive,
    red=negative), plus arrow callouts on a subset of positive cells showing
    their fold-change value. This provides visual proof of which specific cells
    were classified as colocalized.

    Parameters
    ----------
    green_channel : ndarray (Y, X)
        Signal intensity image
    nuclei_labels : ndarray (Y, X)
        Integer label image from StarDist
    measurements_df : DataFrame
        Must contain: label, centroid_y, centroid_x, is_positive, fold_change
    max_annotations : int
        Max number of arrow annotations to draw (avoids clutter)
    figsize : tuple
        Figure size in inches

    Returns
    -------
    Figure
        Matplotlib figure with annotated arrows
    """
    from matplotlib.patches import FancyArrowPatch

    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Normalize green channel
    green_norm = green_channel.astype(float)
    green_norm = (green_norm - green_norm.min()) / (green_norm.max() - green_norm.min() + 1e-8)
    rgb_overlay = np.stack([green_norm, green_norm, green_norm], axis=-1)

    # Draw boundaries
    boundaries = find_boundaries(nuclei_labels, mode='outer')
    positive_labels = set(measurements_df[measurements_df['is_positive']]['label'].values)
    negative_labels = set(measurements_df[~measurements_df['is_positive']]['label'].values)

    boundary_mask = boundaries & (nuclei_labels > 0)
    for label_id in np.unique(nuclei_labels[boundary_mask]):
        if label_id == 0:
            continue
        lmask = boundary_mask & (nuclei_labels == label_id)
        if label_id in positive_labels:
            rgb_overlay[lmask] = [0, 1.0, 0]
        elif label_id in negative_labels:
            rgb_overlay[lmask] = [1.0, 0, 0]

    ax.imshow(rgb_overlay)

    # Annotate top positive cells with arrows
    pos_df = measurements_df[measurements_df['is_positive']].copy()
    if len(pos_df) > 0:
        pos_df = pos_df.sort_values('fold_change', ascending=False)
        annotate_df = pos_df.head(max_annotations)

        h, w = green_channel.shape[:2]
        for _, row in annotate_df.iterrows():
            cy, cx = row['centroid_y'], row['centroid_x']
            fc = row['fold_change']

            # Offset arrow start away from the cell
            offset_y = -40 if cy > h * 0.3 else 40
            offset_x = -40 if cx > w * 0.3 else 40

            ax.annotate(
                f'{fc:.1f}x',
                xy=(cx, cy),
                xytext=(cx + offset_x, cy + offset_y),
                fontsize=7, fontweight='bold', color='white',
                arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7),
            )

    n_pos = len(positive_labels)
    n_neg = len(negative_labels)
    ax.set_title(
        f"Colocalization Proof — {n_pos} positive cells (lime + arrows), {n_neg} negative (red)\n"
        f"Arrows show top {min(max_annotations, n_pos)} positive cells with fold-change values",
        fontsize=13, pad=10)
    ax.axis('off')
    fig.tight_layout()
    return fig


def create_gmm_diagnostic(tissue_pixels, background_diagnostics, figsize=(10, 6)):
    """
    Visualize the GMM background estimation: histogram of tissue intensity
    with fitted Gaussian components overlaid.

    Parameters
    ----------
    tissue_pixels : ndarray (N,)
        Intensity values from tissue-outside-nuclei region
    background_diagnostics : dict
        From ColocalizationAnalyzer.background_diagnostics after GMM fitting.
        Keys: n_components, background_mean, background_std, etc.
    figsize : tuple
        Figure size in inches

    Returns
    -------
    Figure
        Matplotlib figure showing histogram + fitted components
    """
    from scipy.stats import norm

    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Subsample for display if huge
    if len(tissue_pixels) > 500_000:
        rng = np.random.default_rng(42)
        display_pixels = rng.choice(tissue_pixels, size=500_000, replace=False)
    else:
        display_pixels = tissue_pixels

    # Histogram
    counts, bin_edges, patches = ax.hist(
        display_pixels, bins=200, density=True,
        color='gray', alpha=0.6, edgecolor='none', label='Tissue intensity')

    x_range = np.linspace(display_pixels.min(), display_pixels.max(), 500)
    diag = background_diagnostics

    if diag.get('n_components') == 2:
        bg_mean = diag['background_mean']
        bg_std = diag['background_std']
        bg_w = diag['background_weight']
        sig_mean = diag['signal_bleed_mean']
        sig_std = diag['signal_bleed_std']
        sig_w = diag['signal_bleed_weight']

        # Plot components
        bg_curve = bg_w * norm.pdf(x_range, bg_mean, bg_std)
        sig_curve = sig_w * norm.pdf(x_range, sig_mean, sig_std)
        combined = bg_curve + sig_curve

        ax.plot(x_range, bg_curve, 'b-', lw=2,
                label=f'Background (mean={bg_mean:.1f}, w={bg_w:.2f})')
        ax.plot(x_range, sig_curve, 'r-', lw=2,
                label=f'Signal bleed (mean={sig_mean:.1f}, w={sig_w:.2f})')
        ax.plot(x_range, combined, 'k--', lw=1.5, label='Combined fit')

        ax.axvline(bg_mean, color='blue', linestyle=':', alpha=0.7)
        ax.axvline(sig_mean, color='red', linestyle=':', alpha=0.7)

        sep = diag.get('separation', 0)
        conf = diag.get('confidence', 'unknown')
        ax.set_title(
            f"GMM Background Estimation — 2 components\n"
            f"Separation: {sep:.2f} | Confidence: {conf}",
            fontsize=12, pad=10)

    elif diag.get('n_components') == 1:
        bg_mean = diag['background_mean']
        bg_std = diag['background_std']
        curve = norm.pdf(x_range, bg_mean, bg_std)
        ax.plot(x_range, curve, 'b-', lw=2,
                label=f'Single component (mean={bg_mean:.1f})')
        ax.axvline(bg_mean, color='blue', linestyle=':', alpha=0.7)
        ax.set_title(
            "GMM Background Estimation — 1 component (uniform tissue)",
            fontsize=12, pad=10)
    else:
        ax.set_title("GMM Background Estimation — fallback (insufficient data)",
                      fontsize=12, pad=10)

    ax.set_xlabel("Pixel Intensity", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def generate_human_readable_report(summary, background_diagnostics=None,
                                    threshold=2.0, method='fold_change'):
    """
    Generate a plain-English report explaining the colocalization methodology
    and results. Written for non-technical researchers.

    Parameters
    ----------
    summary : dict
        From ColocalizationAnalyzer.get_summary_statistics()
    background_diagnostics : dict or None
        From ColocalizationAnalyzer.background_diagnostics
    threshold : float
        Fold-change threshold used
    method : str
        Classification method used

    Returns
    -------
    str
        Multi-paragraph human-readable report
    """
    total = summary.get('total_cells', 0)
    pos = summary.get('positive_cells', 0)
    neg = summary.get('negative_cells', 0)
    frac = summary.get('positive_fraction', 0)
    bg = summary.get('background_used', 0)
    mean_fc = summary.get('mean_fold_change', 0)
    median_fc = summary.get('median_fold_change', 0)

    lines = []
    lines.append("=" * 70)
    lines.append("COLOCALIZATION ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")

    # --- What we did ---
    lines.append("WHAT WE DID")
    lines.append("-" * 40)
    lines.append(
        "We analyzed a brain tissue section to determine which detected cells "
        "(nuclei) also express a fluorescent signal of interest (e.g., eYFP). "
        "A cell is considered 'positive' (colocalized) if its signal is "
        "significantly brighter than the surrounding tissue background."
    )
    lines.append("")

    # --- How background was determined ---
    lines.append("HOW WE DETERMINED BACKGROUND")
    lines.append("-" * 40)

    diag = background_diagnostics or summary.get('background_diagnostics')
    if diag and diag.get('n_components') == 2:
        lines.append(
            "We used a Gaussian Mixture Model (GMM) to statistically separate "
            "the tissue into two populations: a dim 'background' population and "
            "a brighter 'signal bleed' population. Think of it like sorting a "
            "mixed bag of marbles into two groups by brightness — the dimmer "
            "group represents the natural tissue fluorescence (background), "
            "and the brighter group represents areas with some signal."
        )
        lines.append("")
        lines.append(f"  Background brightness level: {diag['background_mean']:.1f}")
        lines.append(f"  Signal bleed brightness level: {diag.get('signal_bleed_mean', 0):.1f}")
        sep = diag.get('separation', 0)
        conf = diag.get('confidence', 'unknown')
        lines.append(f"  How well-separated the groups are: {sep:.2f} "
                      f"(confidence: {conf})")
        if conf == 'high':
            lines.append(
                "  -> The two groups are clearly distinct, giving us high "
                "confidence in the background estimate."
            )
        elif conf == 'moderate':
            lines.append(
                "  -> The groups overlap somewhat, so the background estimate "
                "is reasonable but should be checked visually."
            )
        else:
            lines.append(
                "  -> The groups are hard to separate. Consider adjusting "
                "imaging parameters or reviewing the tissue mask."
            )
    elif diag and diag.get('n_components') == 1:
        lines.append(
            "We used a Gaussian Mixture Model and found that the tissue has a "
            "uniform brightness — there is one clear background level. This "
            "makes the background estimate straightforward and reliable."
        )
        lines.append(f"  Background brightness level: {diag['background_mean']:.1f}")
    else:
        lines.append(
            f"Background was estimated using the tissue region outside of "
            f"detected cells. The estimated background intensity is {bg:.1f}."
        )
    lines.append("")

    # --- How we classified cells ---
    lines.append("HOW WE CLASSIFIED CELLS")
    lines.append("-" * 40)
    if method == 'fold_change':
        lines.append(
            f"Each cell's average brightness was compared to the background. "
            f"If a cell was at least {threshold:.1f}x brighter than background, "
            f"it was classified as 'positive' (expressing the signal). "
            f"Otherwise, it was classified as 'negative'."
        )
        lines.append("")
        lines.append(
            f"In plain terms: background brightness = {bg:.1f}. "
            f"A cell needed to be at least {bg * threshold:.1f} "
            f"(= {bg:.1f} x {threshold:.1f}) to count as positive."
        )
    elif method == 'absolute':
        lines.append(
            f"Each cell was classified as positive if its average brightness "
            f"exceeded {threshold:.1f} (an absolute intensity cutoff)."
        )
    elif method == 'percentile':
        lines.append(
            f"Cells were ranked by brightness. The top {100 - threshold:.0f}% "
            f"brightest cells were classified as positive."
        )
    lines.append("")

    # --- Results ---
    lines.append("RESULTS")
    lines.append("-" * 40)
    lines.append(f"  Total cells detected:   {total}")
    lines.append(f"  Positive (colocalized): {pos}  ({frac:.1%})")
    lines.append(f"  Negative:               {neg}  ({1 - frac:.1%})")
    lines.append(f"  Average fold-change:    {mean_fc:.2f}x over background")
    lines.append(f"  Median fold-change:     {median_fc:.2f}x over background")
    lines.append("")

    # --- What to look at ---
    lines.append("HOW TO VERIFY THESE RESULTS")
    lines.append("-" * 40)
    lines.append(
        "1. OVERLAY IMAGE: Open the annotated overlay image. Cells outlined "
        "in green (lime) are positive; cells outlined in red are negative. "
        "Arrows point to the brightest positive cells with their fold-change "
        "values. Check that the green-outlined cells visually appear brighter "
        "in the signal channel."
    )
    lines.append(
        "2. FOLD-CHANGE HISTOGRAM: Shows the distribution of cell brightness "
        "relative to background. The vertical dashed line is the threshold. "
        "Green bars (right of line) = positive cells."
    )
    lines.append(
        "3. GMM DIAGNOSTIC PLOT: Shows how background was determined. "
        "The blue curve is the background population, the red curve is signal "
        "bleed. Check that they look like reasonable fits to the histogram."
    )
    lines.append(
        "4. BACKGROUND MASK: Shows which region was used to estimate "
        "background (blue overlay) and which regions were excluded as nuclei "
        "(red overlay). Verify the blue region covers actual tissue."
    )
    lines.append("")
    lines.append("=" * 70)
    lines.append("Report generated by BrainTools colocalization pipeline")
    lines.append("=" * 70)

    return "\n".join(lines)


def save_all_qc_figures(output_dir, green_channel, nuclei_labels, measurements_df,
                        tissue_mask, threshold, background, roi_counts=None,
                        background_diagnostics=None, tissue_pixels=None,
                        summary=None, method='fold_change',
                        prefix="qc", dpi=150):
    """
    Generate and save all QC figures, annotated overlays, and report to disk.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save figures
    green_channel : ndarray
        Signal intensity image
    nuclei_labels : ndarray
        Label image
    measurements_df : DataFrame
        Measurements table (must include is_positive, fold_change, etc.)
    tissue_mask : ndarray
        Tissue mask
    threshold : float
        Fold-change threshold
    background : float
        Background intensity
    roi_counts : list of dict, optional
        ROI summary data
    background_diagnostics : dict, optional
        GMM diagnostics from ColocalizationAnalyzer
    tissue_pixels : ndarray, optional
        Tissue intensity values (for GMM diagnostic plot)
    summary : dict, optional
        Summary statistics from get_summary_statistics()
    method : str
        Classification method used
    prefix : str
        Prefix for saved filenames
    dpi : int
        Resolution for saved images

    Returns
    -------
    list of Path
        Paths to saved files (images + report)
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

    # 2. Annotated overlay with arrows
    fig = create_annotated_overlay(green_channel, nuclei_labels, measurements_df)
    path = output_dir / f"{prefix}_annotated_overlay.png"
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    saved_paths.append(path)

    # 3. Background mask
    fig = create_background_mask_overlay(green_channel, nuclei_labels, tissue_mask)
    path = output_dir / f"{prefix}_background_mask.png"
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    saved_paths.append(path)

    # 4. Fold change histogram
    fig = create_fold_change_histogram(measurements_df, threshold, background)
    path = output_dir / f"{prefix}_fold_change_histogram.png"
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    saved_paths.append(path)

    # 5. Intensity scatter
    fig = create_intensity_scatter(measurements_df, background, threshold)
    path = output_dir / f"{prefix}_intensity_scatter.png"
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    saved_paths.append(path)

    # 6. GMM diagnostic (if data available)
    if background_diagnostics is not None and tissue_pixels is not None:
        fig = create_gmm_diagnostic(tissue_pixels, background_diagnostics)
        path = output_dir / f"{prefix}_gmm_diagnostic.png"
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        saved_paths.append(path)

    # 7. ROI summary (if provided)
    if roi_counts is not None:
        fig = create_roi_summary_bar(roi_counts)
        path = output_dir / f"{prefix}_roi_summary.png"
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        saved_paths.append(path)

    # 8. Human-readable report
    if summary is not None:
        report = generate_human_readable_report(
            summary, background_diagnostics=background_diagnostics,
            threshold=threshold, method=method)
        path = output_dir / f"{prefix}_report.txt"
        path.write_text(report, encoding='utf-8')
        saved_paths.append(path)

    return saved_paths
