"""
Publication-quality visualization for colocalization QC.

Generates fluorescence composite overlays (red channel in R, green channel in G)
with classified cell boundaries, annotation arrows, and diagnostic plots.

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


# ══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════

def _normalize_channel(channel, gamma=0.7, pmin=0.5, pmax=99.5,
                       display_max=None, floor=0):
    """
    Normalize an image channel to 0–1.

    Two modes:
    - If display_max is set: linear map [floor, display_max] → [0, 1] (standard)
    - Otherwise: percentile-based contrast stretch with gamma (legacy)
    """
    img = channel.astype(np.float64)
    if display_max is not None:
        return np.clip((img - floor) / (display_max - floor), 0, 1)
    # Legacy percentile mode
    lo = np.percentile(img, pmin)
    hi = np.percentile(img, pmax)
    if hi - lo < 1e-8:
        return np.zeros_like(img)
    img = np.clip((img - lo) / (hi - lo), 0, 1)
    if gamma != 1.0:
        img = np.power(img, gamma)
    return img


# Display standard constants — canonical source:
# mousebrain.plugin_2d.sliceatlas.core.colocalization
_DISP_RED_FLOOR, _DISP_RED_MAX = 0, 250
_DISP_GRN_FLOOR, _DISP_GRN_MAX = 200, 450


def _make_composite(red_channel=None, green_channel=None, gamma=0.7,
                    use_standard_display=False):
    """
    Create an RGB fluorescence composite from one or both channels.

    Pseudocolor: Magenta (R+B) for red channel, Green for green channel.
    Where both overlap → white. Colorblind-friendly.

    use_standard_display=True uses fixed display ranges instead of
    percentile-based normalization (recommended for dual-channel ENCR data).
    """
    if red_channel is not None and green_channel is not None:
        if use_standard_display:
            r = _normalize_channel(red_channel, display_max=_DISP_RED_MAX,
                                   floor=_DISP_RED_FLOOR)
            g = _normalize_channel(green_channel, display_max=_DISP_GRN_MAX,
                                   floor=_DISP_GRN_FLOOR)
        else:
            r = _normalize_channel(red_channel, gamma=gamma)
            g = _normalize_channel(green_channel, gamma=gamma)
        # Magenta + Green composite (white where both overlap)
        return np.stack([r, g, r], axis=-1)
    elif green_channel is not None:
        if use_standard_display:
            g = _normalize_channel(green_channel, display_max=_DISP_GRN_MAX,
                                   floor=_DISP_GRN_FLOOR)
        else:
            g = _normalize_channel(green_channel, gamma=gamma)
        return np.stack([np.zeros_like(g), g, np.zeros_like(g)], axis=-1)
    elif red_channel is not None:
        if use_standard_display:
            r = _normalize_channel(red_channel, display_max=_DISP_RED_MAX,
                                   floor=_DISP_RED_FLOOR)
        else:
            r = _normalize_channel(red_channel, gamma=gamma)
        return np.stack([r, np.zeros_like(r), r], axis=-1)
    else:
        raise ValueError("At least one channel must be provided")


def _draw_boundaries(rgb, nuclei_labels, measurements_df, thickness=1):
    """
    Draw classified cell boundaries onto an RGB image (in-place).

    Colocalized (positive) cells: bright gold (#FFD700) thick boundary
    Negative cells: dim semi-transparent cyan (#4488AA) thin boundary

    Returns the modified rgb array.
    """
    from scipy import ndimage

    boundaries = find_boundaries(nuclei_labels, mode='outer')

    # Thicken boundaries if requested
    if thickness > 1:
        boundaries = ndimage.binary_dilation(boundaries, iterations=thickness - 1)

    positive_labels = set(measurements_df[measurements_df['is_positive']]['label'].values)
    negative_labels = set(measurements_df[~measurements_df['is_positive']]['label'].values)

    boundary_mask = boundaries & (nuclei_labels > 0)

    for label_id in np.unique(nuclei_labels[boundary_mask]):
        if label_id == 0:
            continue
        lmask = boundary_mask & (nuclei_labels == label_id)

        if label_id in positive_labels:
            # Gold/yellow boundary — bright, stands out on dark background
            rgb[lmask] = [1.0, 0.84, 0.0]  # #FFD700
        elif label_id in negative_labels:
            # Dim cyan — visible but unobtrusive
            alpha = 0.5
            rgb[lmask] = rgb[lmask] * (1 - alpha) + np.array([0.27, 0.53, 0.67]) * alpha

    return rgb


def _dark_style_axes(ax, title='', fontsize=13):
    """Apply dark theme to an axes for fluorescence images."""
    ax.set_facecolor('black')
    ax.set_title(title, fontsize=fontsize, pad=10, color='white', fontweight='bold')
    ax.axis('off')


# ══════════════════════════════════════════════════════════════════════════
# Main overlay images
# ══════════════════════════════════════════════════════════════════════════

def create_overlay_image(green_channel, nuclei_labels, measurements_df,
                         red_channel=None, figsize=(14, 11)):
    """
    Fluorescence composite with classified cell boundaries.

    Shows both channels as a microscope-style composite (red in R, green in G),
    with cell boundaries drawn in gold (positive) or dim cyan (negative).

    Parameters
    ----------
    green_channel : ndarray (Y, X)
        Signal intensity image (e.g., eYFP/488nm)
    nuclei_labels : ndarray (Y, X)
        Integer label image from StarDist (0=background)
    measurements_df : DataFrame
        Must contain columns: label, is_positive
    red_channel : ndarray (Y, X), optional
        Nuclear channel (e.g., 561nm). If None, uses greyscale base.
    figsize : tuple
        Figure size in inches

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    fig = Figure(figsize=figsize, facecolor='black')
    ax = fig.add_subplot(111)

    # Build fluorescence composite
    dual = red_channel is not None and green_channel is not None
    composite = _make_composite(red_channel, green_channel,
                                use_standard_display=dual)

    # Draw boundaries (2px thick for visibility)
    _draw_boundaries(composite, nuclei_labels, measurements_df, thickness=2)

    ax.imshow(composite, interpolation='bilinear')

    n_pos = measurements_df['is_positive'].sum()
    n_neg = (~measurements_df['is_positive']).sum()
    n_total = len(measurements_df)
    pct = 100 * n_pos / n_total if n_total > 0 else 0

    channels = "Red + Green composite" if red_channel is not None else "Green channel"
    _dark_style_axes(ax,
        f"Colocalization — {n_pos} positive (gold), {n_neg} negative (cyan)\n"
        f"{pct:.1f}% colocalized  |  {channels}",
        fontsize=13)

    # Legend in bottom-left
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#FFD700', lw=3, label=f'Colocalized ({n_pos})'),
        Line2D([0], [0], color='#4488AA', lw=2, alpha=0.7, label=f'Negative ({n_neg})'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10,
              facecolor='black', edgecolor='#444444', labelcolor='white',
              framealpha=0.8)

    fig.tight_layout(pad=0.5)
    return fig


def create_annotated_overlay(green_channel, nuclei_labels, measurements_df,
                              red_channel=None, max_annotations=25,
                              figsize=(16, 13)):
    """
    Fluorescence composite with arrow annotations on colocalized cells.

    Shows both channels as composite, with gold boundaries on positive cells,
    plus arrow callouts showing fold-change values on the brightest positives.
    This is the "proof image" — visual evidence of which cells were classified.

    Parameters
    ----------
    green_channel : ndarray (Y, X)
        Signal intensity image
    nuclei_labels : ndarray (Y, X)
        Integer label image from StarDist
    measurements_df : DataFrame
        Must contain: label, centroid_y, centroid_x, is_positive, fold_change
    red_channel : ndarray (Y, X), optional
        Nuclear channel. If None, uses greyscale base.
    max_annotations : int
        Max arrow annotations (avoids clutter)
    figsize : tuple
        Figure size in inches

    Returns
    -------
    Figure
        Matplotlib figure
    """
    fig = Figure(figsize=figsize, facecolor='black')
    ax = fig.add_subplot(111)

    # Build composite and draw boundaries
    dual = red_channel is not None and green_channel is not None
    composite = _make_composite(red_channel, green_channel,
                                use_standard_display=dual)
    _draw_boundaries(composite, nuclei_labels, measurements_df, thickness=2)

    ax.imshow(composite, interpolation='bilinear')

    # Annotate top positive cells with arrows
    pos_df = measurements_df[measurements_df['is_positive']].copy()
    n_pos = len(pos_df)

    if n_pos > 0:
        pos_df = pos_df.sort_values('fold_change', ascending=False)
        annotate_df = pos_df.head(max_annotations)

        h, w = green_channel.shape[:2]
        for _, row in annotate_df.iterrows():
            cy = row.get('centroid_y_base', row.get('centroid_y', 0))
            cx = row.get('centroid_x_base', row.get('centroid_x', 0))
            fc = row['fold_change']

            # Smart offset: push arrow away from image edges
            offset_y = -50 if cy > h * 0.25 else 50
            offset_x = -50 if cx > w * 0.25 else 50

            ax.annotate(
                f'{fc:.1f}x',
                xy=(cx, cy),
                xytext=(cx + offset_x, cy + offset_y),
                fontsize=8, fontweight='bold', color='#FFD700',
                arrowprops=dict(
                    arrowstyle='-|>',
                    color='white',
                    lw=1.5,
                    connectionstyle='arc3,rad=0.15',
                ),
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='black',
                    edgecolor='#FFD700',
                    alpha=0.85,
                    linewidth=1.0,
                ),
            )

    n_neg = (~measurements_df['is_positive']).sum()
    n_total = len(measurements_df)
    pct = 100 * n_pos / n_total if n_total > 0 else 0
    n_shown = min(max_annotations, n_pos)

    _dark_style_axes(ax,
        f"Colocalization Proof — {n_pos}/{n_total} cells positive ({pct:.1f}%)\n"
        f"Arrows: top {n_shown} by fold-change  |  Gold = colocalized, Cyan = negative",
        fontsize=13)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#FFD700', lw=3, label=f'Colocalized ({n_pos})'),
        Line2D([0], [0], color='#4488AA', lw=2, alpha=0.7, label=f'Negative ({n_neg})'),
        Line2D([0], [0], marker='>', color='white', markersize=8,
               markerfacecolor='white', linestyle='None',
               label=f'Annotated ({n_shown})'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10,
              facecolor='black', edgecolor='#444444', labelcolor='white',
              framealpha=0.8)

    fig.tight_layout(pad=0.5)
    return fig


def create_background_mask_overlay(green_channel, nuclei_labels, tissue_mask,
                                    cell_body_dilation=8, figsize=(14, 11)):
    """
    Visualize the three zones used for background estimation:

    1. BLUE: Tissue region (where background is sampled from)
    2. RED: Nuclei (excluded from background)
    3. ORANGE: Cell body exclusion zone (dilated around nuclei — excluded
       because cytoplasm of positive cells would contaminate background)

    The actual background sampling region is: blue AND NOT (red OR orange).

    Parameters
    ----------
    green_channel : ndarray (Y, X)
        Signal intensity image
    nuclei_labels : ndarray (Y, X)
        Integer label image
    tissue_mask : ndarray (Y, X), bool
        True where tissue is estimated
    cell_body_dilation : int
        Number of dilation iterations used for cell body exclusion
    figsize : tuple
        Figure size in inches

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    from scipy import ndimage

    fig = Figure(figsize=figsize, facecolor='black')
    ax = fig.add_subplot(111)

    # Dim greyscale base
    green_norm = _normalize_channel(green_channel, gamma=0.8)
    rgb = np.stack([green_norm * 0.3, green_norm * 0.3, green_norm * 0.3], axis=-1)

    nuclei_mask = nuclei_labels > 0

    # Build cell body exclusion zone (the dilation ring around nuclei)
    if cell_body_dilation > 0:
        cell_body_mask = ndimage.binary_dilation(nuclei_mask, iterations=cell_body_dilation)
        exclusion_ring = cell_body_mask & (~nuclei_mask)  # Ring only (not nucleus itself)
    else:
        cell_body_mask = nuclei_mask
        exclusion_ring = np.zeros_like(nuclei_mask)

    # The actual background sampling region
    sampling_region = tissue_mask & (~cell_body_mask)

    # Layer 1: Tissue sampling region = blue tint
    rgb[sampling_region] = rgb[sampling_region] * 0.4 + np.array([0.15, 0.35, 0.8]) * 0.6

    # Layer 2: Exclusion ring (cell body dilation) = orange tint
    rgb[exclusion_ring] = rgb[exclusion_ring] * 0.3 + np.array([1.0, 0.55, 0.1]) * 0.7

    # Layer 3: Nuclei = red
    rgb[nuclei_mask] = rgb[nuclei_mask] * 0.3 + np.array([1.0, 0.15, 0.15]) * 0.7

    ax.imshow(np.clip(rgb, 0, 1), interpolation='bilinear')

    n_nuclei = np.max(nuclei_labels)
    n_tissue = np.sum(sampling_region)
    n_excluded = np.sum(cell_body_mask)
    _dark_style_axes(ax,
        f"Background Estimation Zones — {n_nuclei} nuclei, "
        f"dilation={cell_body_dilation}px\n"
        f"Blue = sampling region ({n_tissue:,} px), "
        f"Orange = cell body exclusion, Red = nuclei",
        fontsize=12)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2659CC', label='Background sampling region'),
        Patch(facecolor='#FF8C1A', label=f'Cell body exclusion ({cell_body_dilation}px dilation)'),
        Patch(facecolor='#FF2626', label='Nuclei (excluded)'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10,
              facecolor='black', edgecolor='#444444', labelcolor='white',
              framealpha=0.8)

    fig.tight_layout(pad=0.5)
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Diagnostic plots (dark-themed)
# ══════════════════════════════════════════════════════════════════════════

def create_fold_change_histogram(measurements_df, threshold, background, figsize=(9, 5.5)):
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
    fig = Figure(figsize=figsize, facecolor='#1a1a2e')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#16213e')

    fold_changes = measurements_df['fold_change'].values
    is_positive = measurements_df['is_positive'].values

    n_bins = 50
    hist_range = (fold_changes.min(), min(fold_changes.max(), threshold * 5))

    counts, bins, patches = ax.hist(fold_changes, bins=n_bins, range=hist_range,
                                     edgecolor='#333333', linewidth=0.5)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    for patch, bin_center in zip(patches, bin_centers):
        if bin_center >= threshold:
            patch.set_facecolor('#FFD700')
            patch.set_alpha(0.85)
        else:
            patch.set_facecolor('#4488AA')
            patch.set_alpha(0.7)

    ax.axvline(threshold, color='white', linestyle='--', linewidth=2,
               label=f'Threshold ({threshold}x)', alpha=0.9)
    ax.axvline(1.0, color='#666666', linestyle=':', linewidth=1.5,
               label='Fold Change = 1.0')

    n_positive = int(is_positive.sum())
    n_negative = int((~is_positive).sum())
    total = len(is_positive)
    pct_positive = 100 * n_positive / total if total > 0 else 0

    text_str = (f"Colocalized:  {n_positive}  ({pct_positive:.1f}%)\n"
                f"Negative:     {n_negative}\n"
                f"Total:        {total}")
    ax.text(0.97, 0.95, text_str, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='black',
                      edgecolor='#FFD700', alpha=0.85, linewidth=1),
            fontsize=10, fontfamily='monospace', color='white')

    ax.set_xlabel("Fold Change over Background", fontsize=11, color='white')
    ax.set_ylabel("Count", fontsize=11, color='white')
    ax.set_title(f"Fold Change Distribution  (bg={background:.1f}, threshold={threshold}x)",
                 fontsize=12, pad=10, color='white', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, facecolor='black',
              edgecolor='#444444', labelcolor='white', framealpha=0.8)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.15, color='white')
    for spine in ax.spines.values():
        spine.set_color('#444444')

    fig.tight_layout()
    return fig


def create_intensity_scatter(measurements_df, background, threshold, figsize=(9, 6)):
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
    fig = Figure(figsize=figsize, facecolor='#1a1a2e')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#16213e')

    positive_df = measurements_df[measurements_df['is_positive']]
    negative_df = measurements_df[~measurements_df['is_positive']]

    if len(negative_df) > 0:
        ax.scatter(negative_df['area'], negative_df['mean_intensity'],
                  c='#4488AA', alpha=0.4, s=15, edgecolors='none',
                  label=f'Negative (n={len(negative_df)})')

    if len(positive_df) > 0:
        ax.scatter(positive_df['area'], positive_df['mean_intensity'],
                  c='#FFD700', alpha=0.7, s=25, edgecolors='white',
                  linewidth=0.3,
                  label=f'Colocalized (n={len(positive_df)})')

    ax.axhline(background, color='#5599FF', linestyle='--', linewidth=1.5,
               label=f'Background ({background:.1f})', alpha=0.8)

    threshold_intensity = background * threshold
    ax.axhline(threshold_intensity, color='white', linestyle='--', linewidth=1.5,
               label=f'Threshold ({threshold_intensity:.1f})', alpha=0.8)

    ax.set_xlabel("Nucleus Area (pixels)", fontsize=11, color='white')
    ax.set_ylabel("Mean Signal Intensity", fontsize=11, color='white')
    ax.set_title("Intensity vs Size", fontsize=12, pad=10, color='white', fontweight='bold')
    ax.legend(loc='best', fontsize=9, facecolor='black',
              edgecolor='#444444', labelcolor='white', framealpha=0.8)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.15, color='white')
    for spine in ax.spines.values():
        spine.set_color('#444444')

    fig.tight_layout()
    return fig


def create_roi_summary_bar(roi_counts_list, figsize=(9, 5.5)):
    """
    Grouped bar chart of positive/negative counts per ROI.

    Parameters
    ----------
    roi_counts_list : list of dict
        Each dict has keys: roi, total, positive, negative, fraction
    figsize : tuple
        Figure size in inches

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    fig = Figure(figsize=figsize, facecolor='#1a1a2e')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#16213e')

    roi_data = [r for r in roi_counts_list if r['roi'] != 'TOTAL']

    if len(roi_data) == 0:
        ax.text(0.5, 0.5, 'No ROI data available',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='white')
        return fig

    roi_names = [r['roi'] for r in roi_data]
    positive_counts = [r['positive'] for r in roi_data]
    negative_counts = [r['negative'] for r in roi_data]
    fractions = [r['fraction'] for r in roi_data]

    x = np.arange(len(roi_names))
    width = 0.35

    bars_pos = ax.bar(x - width/2, positive_counts, width,
                      label='Colocalized', color='#FFD700', alpha=0.85,
                      edgecolor='#333333')
    bars_neg = ax.bar(x + width/2, negative_counts, width,
                      label='Negative', color='#4488AA', alpha=0.75,
                      edgecolor='#333333')

    for bar, frac in zip(bars_pos, fractions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{frac:.1%}', ha='center', va='bottom', fontsize=9,
                color='white', fontweight='bold')

    ax.set_xlabel('ROI', fontsize=11, color='white')
    ax.set_ylabel('Cell Count', fontsize=11, color='white')
    ax.set_title('ROI Cell Counts', fontsize=12, pad=10, color='white', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(roi_names, rotation=45, ha='right', color='white')
    ax.legend(loc='upper right', fontsize=9, facecolor='black',
              edgecolor='#444444', labelcolor='white', framealpha=0.8)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.15, color='white', axis='y')
    for spine in ax.spines.values():
        spine.set_color('#444444')

    fig.tight_layout()
    return fig


def create_background_surface_plot(background_surface, nuclei_labels=None,
                                    figsize=(10, 8)):
    """
    Heatmap of the interpolated 2D local background surface.

    Shows how background intensity varies spatially across the tissue,
    with optional nucleus centroid overlay.

    Parameters
    ----------
    background_surface : ndarray (Y, X)
        2D background intensity array from estimate_local_background()
    nuclei_labels : ndarray (Y, X), optional
        Label image — if provided, nucleus centroids are overlaid as dots
    figsize : tuple
        Figure size in inches

    Returns
    -------
    Figure
        Matplotlib figure
    """
    from scipy import ndimage

    fig = Figure(figsize=figsize, facecolor='#1a1a2e')
    ax = fig.add_subplot(111)

    # Plot the background surface as a heatmap
    im = ax.imshow(background_surface, cmap='inferno', interpolation='bilinear')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Background Intensity', fontsize=10, color='white')
    cbar.ax.tick_params(colors='white')

    # Overlay nucleus centroids if labels provided
    if nuclei_labels is not None:
        unique_labels = np.unique(nuclei_labels)
        unique_labels = unique_labels[unique_labels > 0]
        if len(unique_labels) > 0:
            centroids = ndimage.center_of_mass(
                nuclei_labels > 0, nuclei_labels, unique_labels
            )
            cy = [c[0] for c in centroids]
            cx = [c[1] for c in centroids]
            ax.scatter(cx, cy, s=3, c='white', alpha=0.4, edgecolors='none',
                      label=f'{len(unique_labels)} nuclei')
            ax.legend(loc='lower right', fontsize=9, facecolor='black',
                      edgecolor='#444444', labelcolor='white', framealpha=0.8)

    bg_min = background_surface.min()
    bg_max = background_surface.max()
    bg_mean = background_surface.mean()
    bg_range = bg_max - bg_min
    variation_pct = 100 * bg_range / bg_mean if bg_mean > 0 else 0

    ax.set_title(
        f"Local Background Surface\n"
        f"Range: {bg_min:.1f} — {bg_max:.1f}  |  "
        f"Mean: {bg_mean:.1f}  |  Variation: {variation_pct:.1f}%",
        fontsize=12, pad=10, color='white', fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444444')

    fig.tight_layout()
    return fig


def create_gmm_diagnostic(tissue_pixels, background_diagnostics, figsize=(10, 6)):
    """
    GMM background estimation diagnostic: histogram with fitted components.

    Parameters
    ----------
    tissue_pixels : ndarray (N,)
        Intensity values from tissue-outside-cells region
    background_diagnostics : dict
        From ColocalizationAnalyzer.background_diagnostics
    figsize : tuple
        Figure size in inches

    Returns
    -------
    Figure
        Matplotlib figure
    """
    from scipy.stats import norm

    fig = Figure(figsize=figsize, facecolor='#1a1a2e')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#16213e')

    if len(tissue_pixels) > 500_000:
        rng = np.random.default_rng(42)
        display_pixels = rng.choice(tissue_pixels, size=500_000, replace=False)
    else:
        display_pixels = tissue_pixels

    ax.hist(display_pixels, bins=200, density=True,
            color='#555577', alpha=0.6, edgecolor='none', label='Tissue intensity')

    x_range = np.linspace(display_pixels.min(), display_pixels.max(), 500)
    diag = background_diagnostics

    if diag.get('n_components') == 2:
        bg_mean = diag['background_mean']
        bg_std = diag['background_std']
        bg_w = diag['background_weight']
        sig_mean = diag['signal_bleed_mean']
        sig_std = diag['signal_bleed_std']
        sig_w = diag['signal_bleed_weight']

        bg_curve = bg_w * norm.pdf(x_range, bg_mean, bg_std)
        sig_curve = sig_w * norm.pdf(x_range, sig_mean, sig_std)
        combined = bg_curve + sig_curve

        ax.plot(x_range, bg_curve, color='#5599FF', lw=2.5,
                label=f'Background (mean={bg_mean:.1f}, w={bg_w:.2f})')
        ax.plot(x_range, sig_curve, color='#FF6644', lw=2.5,
                label=f'Signal bleed (mean={sig_mean:.1f}, w={sig_w:.2f})')
        ax.plot(x_range, combined, 'w--', lw=1.5, alpha=0.7, label='Combined fit')

        ax.axvline(bg_mean, color='#5599FF', linestyle=':', alpha=0.7)
        ax.axvline(sig_mean, color='#FF6644', linestyle=':', alpha=0.7)

        sep = diag.get('separation', 0)
        conf = diag.get('confidence', 'unknown')
        title = (f"GMM Background Estimation — 2 components\n"
                 f"Separation: {sep:.2f} | Confidence: {conf}")

    elif diag.get('n_components') == 1:
        bg_mean = diag['background_mean']
        bg_std = diag['background_std']
        curve = norm.pdf(x_range, bg_mean, bg_std)
        ax.plot(x_range, curve, color='#5599FF', lw=2.5,
                label=f'Single component (mean={bg_mean:.1f})')
        ax.axvline(bg_mean, color='#5599FF', linestyle=':', alpha=0.7)
        title = "GMM Background Estimation — 1 component (uniform tissue)"
    else:
        title = "GMM Background Estimation — fallback (insufficient data)"

    ax.set_xlabel("Pixel Intensity", fontsize=11, color='white')
    ax.set_ylabel("Density", fontsize=11, color='white')
    ax.set_title(title, fontsize=12, pad=10, color='white', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, facecolor='black',
              edgecolor='#444444', labelcolor='white', framealpha=0.8)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.15, color='white')
    for spine in ax.spines.values():
        spine.set_color('#444444')

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════

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

    lines.append("WHAT WE DID")
    lines.append("-" * 40)
    lines.append(
        "We analyzed a brain tissue section to determine which detected cells "
        "(nuclei) also express a fluorescent signal of interest (e.g., eYFP). "
        "A cell is considered 'positive' (colocalized) if its signal is "
        "significantly brighter than the surrounding tissue background."
    )
    lines.append("")

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
    elif method == 'area_fraction':
        lines.append(
            f"Each cell was examined pixel-by-pixel. For each nucleus, we "
            f"checked what fraction of its pixels were at least {threshold:.1f}x "
            f"brighter than background ({bg:.1f}). A cell was classified as "
            f"positive only if a sufficient fraction of its area exceeded this "
            f"brightness cutoff."
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

    lines.append("RESULTS")
    lines.append("-" * 40)
    lines.append(f"  Total cells detected:   {total}")
    lines.append(f"  Positive (colocalized): {pos}  ({frac:.1%})")
    lines.append(f"  Negative:               {neg}  ({1 - frac:.1%})")
    lines.append(f"  Average fold-change:    {mean_fc:.2f}x over background")
    lines.append(f"  Median fold-change:     {median_fc:.2f}x over background")
    lines.append("")

    lines.append("HOW TO VERIFY THESE RESULTS")
    lines.append("-" * 40)
    lines.append(
        "1. COMPOSITE OVERLAY: Open the overlay image. It shows the red (nuclear) "
        "and green (signal) channels as a fluorescence composite. Cells outlined "
        "in gold are positive (colocalized); cells with dim cyan outlines are "
        "negative. Check that gold-outlined cells visually appear green/yellow."
    )
    lines.append(
        "2. ANNOTATED OVERLAY: Same as above, with arrows pointing to the "
        "brightest positive cells and their fold-change values."
    )
    lines.append(
        "3. FOLD-CHANGE HISTOGRAM: Distribution of cell brightness relative to "
        "background. Gold bars (right of threshold line) = positive cells."
    )
    lines.append(
        "4. GMM DIAGNOSTIC: Shows how background was determined. Blue curve = "
        "background population, red = signal bleed."
    )
    lines.append(
        "5. BACKGROUND MASK: Shows three zones — blue (where background was "
        "sampled), orange (cell body exclusion ring), red (nuclei). Verify the "
        "blue region covers actual tissue between cells."
    )
    lines.append("")
    lines.append("=" * 70)
    lines.append("Report generated by BrainTools colocalization pipeline")
    lines.append("=" * 70)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# Master save function
# ══════════════════════════════════════════════════════════════════════════

def save_all_qc_figures(output_dir, green_channel, nuclei_labels, measurements_df,
                        tissue_mask, threshold, background, roi_counts=None,
                        background_diagnostics=None, tissue_pixels=None,
                        summary=None, method='fold_change',
                        red_channel=None, cell_body_dilation=8,
                        prefix="qc", dpi=200):
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
    red_channel : ndarray, optional
        Nuclear channel for fluorescence composite
    cell_body_dilation : int
        Cell body dilation used in background estimation
    prefix : str
        Prefix for saved filenames
    dpi : int
        Resolution for saved images (default 200 for publication quality)

    Returns
    -------
    list of Path
        Paths to saved files (images + report)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    # 1. Composite overlay with classified boundaries
    fig = create_overlay_image(green_channel, nuclei_labels, measurements_df,
                               red_channel=red_channel)
    path = output_dir / f"{prefix}_overlay.png"
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    saved_paths.append(path)

    # 2. Annotated overlay with arrows on colocalized cells
    fig = create_annotated_overlay(green_channel, nuclei_labels, measurements_df,
                                   red_channel=red_channel)
    path = output_dir / f"{prefix}_annotated_overlay.png"
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    saved_paths.append(path)

    # 3. Background estimation zones (tissue / exclusion ring / nuclei)
    fig = create_background_mask_overlay(green_channel, nuclei_labels, tissue_mask,
                                         cell_body_dilation=cell_body_dilation)
    path = output_dir / f"{prefix}_background_mask.png"
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    saved_paths.append(path)

    # 4. Fold change histogram
    fig = create_fold_change_histogram(measurements_df, threshold, background)
    path = output_dir / f"{prefix}_fold_change_histogram.png"
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    saved_paths.append(path)

    # 5. Intensity scatter
    fig = create_intensity_scatter(measurements_df, background, threshold)
    path = output_dir / f"{prefix}_intensity_scatter.png"
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    saved_paths.append(path)

    # 6. GMM diagnostic (if data available)
    if background_diagnostics is not None and tissue_pixels is not None:
        fig = create_gmm_diagnostic(tissue_pixels, background_diagnostics)
        path = output_dir / f"{prefix}_gmm_diagnostic.png"
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        saved_paths.append(path)

    # 7. ROI summary (if provided)
    if roi_counts is not None:
        fig = create_roi_summary_bar(roi_counts)
        path = output_dir / f"{prefix}_roi_summary.png"
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
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
