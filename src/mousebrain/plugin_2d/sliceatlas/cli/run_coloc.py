"""
run_coloc.py - CLI for quick detection + colocalization on a single ND2/TIFF file.

Runs the full pipeline and pops up a matplotlib figure with results.
Much faster iteration than launching napari every time.

Usage:
    python -m brainslice.cli.run_coloc path/to/image.nd2
    python -m brainslice.cli.run_coloc path/to/image.nd2 --red-ch 1 --green-ch 0
    python -m brainslice.cli.run_coloc path/to/image.nd2 --method percentile --percentile 97
    python -m brainslice.cli.run_coloc path/to/image.nd2 --no-hysteresis --min-area 50
    python -m brainslice.cli.run_coloc path/to/image.nd2 --save output_dir/
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Support both brainslice (standalone) and mousebrain (monorepo) installs
try:
    from brainslice.core import io as _io_mod
    _PKG = 'brainslice'
except ImportError:
    from mousebrain.plugin_2d.sliceatlas.core import io as _io_mod
    _PKG = 'mousebrain'


def _import_core(module_name):
    """Import a core module from whichever package is available."""
    if _PKG == 'brainslice':
        import importlib
        return importlib.import_module(f'brainslice.core.{module_name}')
    else:
        import importlib
        return importlib.import_module(
            f'mousebrain.plugin_2d.sliceatlas.core.{module_name}'
        )


def run_pipeline(args):
    """Run detection + colocalization and show results."""
    t0 = time.time()

    # ── Load image ──
    print(f"Loading: {args.file}")
    io_mod = _import_core('io')
    load_image = io_mod.load_image
    guess_channel_roles = io_mod.guess_channel_roles
    data, metadata = load_image(args.file)
    t_load = time.time() - t0
    print(f"  Loaded in {t_load:.1f}s: shape={data.shape}, dtype={data.dtype}")
    print(f"  Channels: {metadata.get('channels', ['?'])}")
    if 'voxel_size_um' in metadata:
        vs = metadata['voxel_size_um']
        print(f"  Pixel size: {vs.get('x', '?')} um/px")

    # ── Auto-detect or use specified channels ──
    roles = guess_channel_roles(metadata)
    red_ch = args.red_ch if args.red_ch is not None else roles['nuclear']
    green_ch = args.green_ch if args.green_ch is not None else roles['signal']
    ch_names = metadata.get('channels', [])
    red_name = ch_names[red_ch] if red_ch < len(ch_names) else f"ch{red_ch}"
    green_name = ch_names[green_ch] if green_ch < len(ch_names) else f"ch{green_ch}"
    print(f"  Nuclear (red): ch{red_ch} ({red_name})")
    print(f"  Signal (green): ch{green_ch} ({green_name})")

    red_image = data[red_ch]
    green_image = data[green_ch]

    # ── Detect nuclei ──
    print(f"\nDetecting nuclei (threshold, method={args.method})...")
    det_mod = _import_core('detection')
    detect_by_threshold = det_mod.detect_by_threshold
    t1 = time.time()

    labels, details = detect_by_threshold(
        red_image,
        method=args.method,
        percentile=args.percentile,
        manual_threshold=args.manual_threshold,
        min_area=args.min_area,
        max_area=args.max_area,
        opening_radius=args.opening_radius,
        closing_radius=args.closing_radius,
        fill_holes=not args.no_fill_holes,
        split_touching=args.split_touching,
        split_footprint_size=args.split_footprint,
        gaussian_sigma=args.gaussian_sigma,
        use_hysteresis=not args.no_hysteresis,
        hysteresis_low_fraction=args.hysteresis_low,
        min_solidity=args.min_solidity,
        min_circularity=args.min_circularity,
    )
    t_detect = time.time() - t1

    n_nuclei = details['filtered_count']
    print(f"  Found {n_nuclei} nuclei in {t_detect:.1f}s")
    print(f"  Threshold ({details['method']}): {details['threshold']:.1f}")
    if details.get('use_hysteresis'):
        print(f"  Hysteresis low: {details['threshold_low']:.1f}")
    print(f"  Raw: {details['raw_count']} -> Filtered: {n_nuclei} "
          f"(removed {details['removed_by_size']} by size)")
    if details.get('n_watershed_splits', 0) > 0:
        print(f"  Watershed splits: {details['n_watershed_splits']}")
    if details.get('removed_by_morphology', 0) > 0:
        print(f"  Removed by morphology: {details['removed_by_morphology']}")

    if n_nuclei == 0:
        print("\nNo nuclei found. Try lowering threshold or min_area.")
        return

    # ── Branch: dual or single channel ──
    if args.dual:
        classified, summary, coloc_metrics = _run_dual_pipeline(
            red_image, green_image, labels, args, t0, t_load, t_detect, details,
        )
    else:
        classified, summary, coloc_metrics = _run_single_pipeline(
            red_image, green_image, labels, args, t0, t_load, t_detect, details,
        )

    # ── Visualize ──
    print("\nGenerating visualization...")
    if args.dual:
        fig = _make_dual_results_figure(
            red_image, green_image, labels, classified, summary,
            details, args,
        )
    else:
        fig = _make_results_figure(
            red_image, green_image, labels, classified, summary,
            details, coloc_metrics, args,
        )

    # Always save a PNG next to the input file for quick review
    png_path = Path(args.file).with_suffix('.coloc_result.png')
    fig.savefig(str(png_path), dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  Saved figure: {png_path}")

    # Show interactively unless --no-show
    if not args.no_show:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.show()

    # ── Save ──
    if args.save:
        _save_results(args.save, classified, summary, labels, args, fig)


def _run_single_pipeline(red_image, green_image, labels, args, t0, t_load, t_detect, details):
    """Original single-channel colocalization pipeline."""
    n_nuclei = details['filtered_count']

    print(f"\nRunning colocalization (soma_dilation={args.soma_dilation}px)...")
    coloc_mod = _import_core('colocalization')
    ColocalizationAnalyzer = coloc_mod.ColocalizationAnalyzer
    compute_colocalization_metrics = coloc_mod.compute_colocalization_metrics
    t2 = time.time()

    analyzer = ColocalizationAnalyzer()

    # Background estimation
    background = analyzer.estimate_background(
        green_image, labels, dilation_iterations=args.bg_dilation,
    )

    # Tissue mask
    tissue_mask = analyzer.estimate_tissue_mask(labels, args.bg_dilation)

    # Measure intensities — nuclear by default, soma if requested
    if args.soma_dilation > 0:
        measurements = analyzer.measure_soma_intensities(
            green_image, labels, soma_dilation=args.soma_dilation,
        )
    else:
        measurements = analyzer.measure_nuclei_intensities(green_image, labels)

    # Classify
    classified = analyzer.classify_positive_negative(
        measurements, background,
        method=args.coloc_method,
        threshold=args.coloc_threshold,
    )

    summary = analyzer.get_summary_statistics(classified)

    # Print adaptive diagnostics if available
    if summary.get('adaptive_diagnostics'):
        ad = summary['adaptive_diagnostics']
        print(f"  Adaptive method: {ad['method_used']}")
        if ad['method_used'] == 'gmm_2component':
            print(f"    Negative: mean={ad['negative_mean']:.1f}, "
                  f"weight={ad['negative_weight']:.0%}")
            print(f"    Positive: mean={ad['positive_mean']:.1f}, "
                  f"weight={ad['positive_weight']:.0%}")
            print(f"    Separation: {ad['separation']:.2f}, "
                  f"threshold: {ad['adaptive_threshold']:.1f}")
        elif ad.get('reason'):
            print(f"    Reason: {ad['reason']}")

    # Manders/Pearson
    coloc_metrics = compute_colocalization_metrics(
        red_image=red_image,
        green_image=green_image,
        nuclei_labels=labels,
        background_green=summary['background_used'],
        tissue_mask=tissue_mask,
        soma_dilation=args.soma_dilation,
    )
    summary['coloc_metrics'] = coloc_metrics

    t_coloc = time.time() - t2
    t_total = time.time() - t0

    # ── Print results ──
    print(f"\n{'='*50}")
    print(f"RESULTS: {Path(args.file).name}")
    print(f"{'='*50}")
    print(f"  Nuclei detected: {n_nuclei}")
    print(f"  Positive: {summary['positive_cells']} "
          f"({summary['positive_fraction']*100:.1f}%)")
    print(f"  Negative: {summary['negative_cells']}")
    print(f"  Background: {summary['background_used']:.1f}")
    print(f"  Mean fold-change: {summary['mean_fold_change']:.2f}")
    print(f"\n  Validation metrics:")
    print(f"    Pearson r:  {coloc_metrics['pearson_r']:.4f}")
    print(f"    Manders M1: {coloc_metrics['manders_m1']:.4f} "
          f"(red in green regions)")
    print(f"    Manders M2: {coloc_metrics['manders_m2']:.4f} "
          f"(green in red ROIs)")
    print(f"\n  Timing: load={t_load:.1f}s, detect={t_detect:.1f}s, "
          f"coloc={t_coloc:.1f}s, total={t_total:.1f}s")

    return classified, summary, coloc_metrics


def _run_dual_pipeline(red_image, green_image, labels, args, t0, t_load, t_detect, details):
    """Dual-channel colocalization pipeline."""
    n_nuclei = details['filtered_count']

    print(f"\nRunning DUAL-CHANNEL colocalization...")
    print(f"  Ch1 (red):  soma_dilation={args.ch1_soma_dilation}px, "
          f"threshold={args.ch1_threshold}x, bg_dilation={args.ch1_bg_dilation}")
    print(f"  Ch2 (green): soma_dilation={args.ch2_soma_dilation}px, "
          f"threshold={args.ch2_threshold}x, bg_dilation={args.ch2_bg_dilation}")

    coloc_mod = _import_core('colocalization')
    analyze_dual = coloc_mod.analyze_dual_colocalization
    t2 = time.time()

    classified, summary = analyze_dual(
        signal_image_1=red_image,
        signal_image_2=green_image,
        nuclei_labels=labels,
        threshold_method_1='fold_change',
        threshold_value_1=args.ch1_threshold,
        cell_body_dilation_1=args.ch1_bg_dilation,
        soma_dilation_1=args.ch1_soma_dilation,
        threshold_method_2='fold_change',
        threshold_value_2=args.ch2_threshold,
        cell_body_dilation_2=args.ch2_bg_dilation,
        soma_dilation_2=args.ch2_soma_dilation,
        ch1_name='red',
        ch2_name='green',
    )

    t_coloc = time.time() - t2
    t_total = time.time() - t0

    # ── Print results ──
    print(f"\n{'='*50}")
    print(f"DUAL-CHANNEL RESULTS: {Path(args.file).name}")
    print(f"{'='*50}")
    print(f"  Nuclei detected:  {summary['total_nuclei']}")
    print(f"  Red+ (mCherry):   {summary['n_red_positive']} "
          f"({summary['fraction_red']*100:.1f}%)")
    print(f"  Green+ (eYFP):    {summary['n_green_positive']} "
          f"({summary['fraction_green']*100:.1f}%)")
    print(f"  Dual+ (both):     {summary['n_dual']} "
          f"({summary['fraction_dual']*100:.1f}%)")
    print(f"  Red-only:         {summary['n_red_only']}")
    print(f"  Green-only:       {summary['n_green_only']} "
          f"{'(expected ~0)' if summary['n_green_only'] > 0 else ''}")
    print(f"  Neither:          {summary['n_neither']}")
    print(f"  Background red:   {summary['background_red']:.1f}")
    print(f"  Background green: {summary['background_green']:.1f}")
    print(f"\n  Timing: load={t_load:.1f}s, detect={t_detect:.1f}s, "
          f"coloc={t_coloc:.1f}s, total={t_total:.1f}s")

    return classified, summary, None


def _make_results_figure(red_image, green_image, labels, measurements,
                         summary, det_details, coloc_metrics, args):
    """Build a matplotlib figure with detection + colocalization results.

    Layout:
    - Top row: Full image overview (red channel, composite, metrics text)
    - Bottom row: Zoomed panels around individual nuclei (up to 6),
      each showing red + green + composite with boundary outline
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from skimage.segmentation import find_boundaries
    from skimage.measure import regionprops

    # Normalize channels for display using shared standard
    from mousebrain.plugin_2d.sliceatlas.core.colocalization import (
        normalize_for_display, DISPLAY_RED_FLOOR, DISPLAY_RED_MAX,
        DISPLAY_RED_GAMMA, DISPLAY_GRN_FLOOR, DISPLAY_GRN_MAX,
        DISPLAY_GRN_GAMMA,
    )

    r = normalize_for_display(red_image, DISPLAY_RED_MAX, DISPLAY_RED_FLOOR, DISPLAY_RED_GAMMA)
    g = normalize_for_display(green_image, DISPLAY_GRN_MAX, DISPLAY_GRN_FLOOR, DISPLAY_GRN_GAMMA)
    # Magenta + Green composite (white where both overlap)
    composite = np.stack([r, g, r], axis=-1)

    # Find nucleus boundaries
    boundaries = find_boundaries(labels, mode='outer')

    # Classify labels
    positive_labels = set()
    negative_labels = set()
    label_fc = {}  # fold-change per label
    label_prob = {}  # posterior probability per label (adaptive method)
    if measurements is not None and len(measurements) > 0:
        for _, row in measurements.iterrows():
            lbl = int(row['label'])
            if row.get('is_positive', False):
                positive_labels.add(lbl)
            else:
                negative_labels.add(lbl)
            label_fc[lbl] = row.get('fold_change', 0)
            if 'positive_probability' in row.index:
                label_prob[lbl] = row['positive_probability']

    # Per-label colored boundaries
    pos_boundary = np.zeros(labels.shape, dtype=bool)
    neg_boundary = np.zeros(labels.shape, dtype=bool)
    for lbl in positive_labels:
        pos_boundary |= find_boundaries(labels == lbl, mode='outer')
    for lbl in negative_labels:
        neg_boundary |= find_boundaries(labels == lbl, mode='outer')

    # Get nucleus regions for zoom panels
    props = regionprops(labels)
    # Sort by area (largest first) to show most interesting nuclei
    props.sort(key=lambda p: p.area, reverse=True)
    n_zoom = min(6, len(props))
    zoom_pad = 40  # pixels of padding around each nucleus

    # ── Figure layout ──
    # Top row: 3 panels (overview)
    # Bottom row: up to 6 zoom panels (3 cols x 2 rows of triplets)
    n_zoom_rows = max(1, (n_zoom + 2) // 3)  # how many rows of zooms
    fig = plt.figure(figsize=(18, 6 + 4 * n_zoom_rows))
    fig.patch.set_facecolor('#1a1a1a')
    fname = Path(args.file).stem

    # Top row: overview panels
    gs_top = fig.add_gridspec(1, 3, top=0.98, bottom=0.55 if n_zoom > 0 else 0.02,
                              left=0.02, right=0.98, wspace=0.05)

    h, w = labels.shape

    # Panel 1: Red channel with white nucleus outlines
    ax1 = fig.add_subplot(gs_top[0, 0])
    red_rgb = np.stack([r, np.zeros_like(r), r], axis=-1)
    ax1.imshow(np.clip(red_rgb, 0, 1))
    outline_overlay = np.zeros((*labels.shape, 4))
    outline_overlay[boundaries] = [1, 1, 1, 0.9]
    ax1.imshow(outline_overlay)
    # Draw rectangles around zoom regions
    for i, prop in enumerate(props[:n_zoom]):
        cy, cx = prop.centroid
        y0 = max(0, int(cy) - zoom_pad)
        x0 = max(0, int(cx) - zoom_pad)
        sz = 2 * zoom_pad
        rect = Rectangle((x0, y0), sz, sz, linewidth=1,
                          edgecolor='cyan', facecolor='none', linestyle='--')
        ax1.add_patch(rect)
        ax1.text(x0 + 2, y0 - 2, str(i + 1), color='cyan', fontsize=8,
                 va='bottom', fontweight='bold')
    ax1.set_title(f"Detection: {labels.max()} nuclei", color='white', fontsize=12)
    thresh_str = f"{det_details['method']}: {det_details['threshold']:.0f}"
    if det_details.get('use_hysteresis'):
        thresh_str += f" (hyst low: {det_details['threshold_low']:.0f})"
    ax1.text(0.02, 0.02, thresh_str, transform=ax1.transAxes,
             color='#aaa', fontsize=9, va='bottom')
    ax1.axis('off')

    # Panel 2: Composite with pos/neg outlines
    ax2 = fig.add_subplot(gs_top[0, 1])
    ax2.imshow(np.clip(composite, 0, 1))
    coloc_overlay = np.zeros((*labels.shape, 4))
    coloc_overlay[pos_boundary] = [0, 1, 0, 0.9]
    coloc_overlay[neg_boundary] = [1, 0.2, 0.2, 0.9]
    ax2.imshow(coloc_overlay)
    n_pos = summary['positive_cells']
    n_neg = summary['negative_cells']
    frac = summary['positive_fraction'] * 100
    ax2.set_title(
        f"Colocalization: {n_pos} pos ({frac:.0f}%) / {n_neg} neg",
        color='white', fontsize=12,
    )
    ax2.axis('off')

    # Panel 3: Metrics text
    ax3 = fig.add_subplot(gs_top[0, 2])
    ax3.set_facecolor('#1a1a1a')
    ax3.axis('off')
    # Build adaptive diagnostics text
    adapt_text = ""
    ad = summary.get('adaptive_diagnostics')
    if ad:
        if ad['method_used'] == 'gmm_2component':
            adapt_text = (
                f"  GMM: neg={ad['negative_mean']:.0f} "
                f"({ad['negative_weight']:.0%}), "
                f"pos={ad['positive_mean']:.0f} "
                f"({ad['positive_weight']:.0%})\n"
                f"  Separation: {ad['separation']:.1f}, "
                f"thresh: {ad['adaptive_threshold']:.0f}\n"
            )
        else:
            adapt_text = f"  Fallback: {ad.get('reason', 'N/A')}\n"

    measure_type = 'soma' if args.soma_dilation > 0 else 'nuclear'
    metrics_text = (
        f"File: {fname}\n\n"
        f"Detection\n"
        f"  Method: threshold ({det_details['method']})\n"
        f"  Hysteresis: {'ON' if det_details.get('use_hysteresis') else 'OFF'}\n"
        f"  Nuclei: {labels.max()}\n"
        f"  Raw: {det_details['raw_count']} -> {det_details['filtered_count']}\n"
        f"  Min/Max area: {args.min_area}-{args.max_area} px\n\n"
        f"Colocalization ({args.coloc_method})\n"
        f"  Positive: {n_pos} / {n_pos + n_neg} ({frac:.1f}%)\n"
        f"  Background: {summary['background_used']:.1f}\n"
        f"  Mean fold-change: {summary['mean_fold_change']:.2f}\n"
        f"  Measurement: {measure_type}\n"
        f"{adapt_text}\n"
        f"Validation\n"
        f"  Pearson r:  {coloc_metrics['pearson_r']:.4f}\n"
        f"  Manders M1: {coloc_metrics['manders_m1']:.4f}\n"
        f"  Manders M2: {coloc_metrics['manders_m2']:.4f}\n"
    )
    ax3.text(0.05, 0.95, metrics_text, transform=ax3.transAxes,
             color='white', fontsize=10, fontfamily='monospace',
             va='top', ha='left')

    # ── Bottom rows: Zoomed nucleus panels ──
    if n_zoom > 0:
        # Each zoom gets 3 sub-panels: red, green, composite
        n_cols = min(n_zoom, 3)
        n_rows = (n_zoom + n_cols - 1) // n_cols
        gs_bot = fig.add_gridspec(
            n_rows, n_cols * 3,
            top=0.48, bottom=0.02, left=0.02, right=0.98,
            wspace=0.08, hspace=0.25,
        )

        for i, prop in enumerate(props[:n_zoom]):
            row = i // n_cols
            col = (i % n_cols) * 3

            cy, cx = int(prop.centroid[0]), int(prop.centroid[1])
            y0 = max(0, cy - zoom_pad)
            y1 = min(h, cy + zoom_pad)
            x0 = max(0, cx - zoom_pad)
            x1 = min(w, cx + zoom_pad)

            lbl = prop.label
            is_pos = lbl in positive_labels
            fc = label_fc.get(lbl, 0)

            # Crop regions
            r_crop = r[y0:y1, x0:x1]
            g_crop = g[y0:y1, x0:x1]
            comp_crop = composite[y0:y1, x0:x1]

            # Label mask for vector contour overlay
            lbl_crop = labels[y0:y1, x0:x1]
            contour_mask = (lbl_crop == lbl).astype(np.float64)
            contour_color = 'lime' if is_pos else '#ff4444'

            # Red channel zoom — clean image + vector contour
            ax_r = fig.add_subplot(gs_bot[row, col])
            red_crop = np.stack([r_crop, np.zeros_like(r_crop), r_crop], axis=-1)
            ax_r.imshow(np.clip(red_crop, 0, 1))
            ax_r.contour(contour_mask, levels=[0.5], linewidths=0.8,
                         colors=['white'], antialiased=True)
            ax_r.set_title(f"#{i+1} Red", color='white', fontsize=9)
            ax_r.axis('off')

            # Green channel zoom — clean image + vector contour
            ax_g = fig.add_subplot(gs_bot[row, col + 1])
            grn_crop = np.stack([np.zeros_like(g_crop), g_crop, np.zeros_like(g_crop)], axis=-1)
            ax_g.imshow(np.clip(grn_crop, 0, 1))
            ax_g.contour(contour_mask, levels=[0.5], linewidths=0.8,
                         colors=[contour_color], antialiased=True)
            ax_g.set_title(f"Green", color='white', fontsize=9)
            ax_g.axis('off')

            # Composite zoom — clean image + vector contour
            ax_c = fig.add_subplot(gs_bot[row, col + 2])
            ax_c.imshow(np.clip(comp_crop, 0, 1))
            ax_c.contour(contour_mask, levels=[0.5], linewidths=0.8,
                         colors=[contour_color], antialiased=True)
            status = "POS" if is_pos else "NEG"
            status_color = 'lime' if is_pos else '#ff4444'
            # Show probability if available (adaptive method), otherwise fold-change
            prob = label_prob.get(lbl)
            if prob is not None:
                ax_c.set_title(f"{status} P={prob:.2f}", color=status_color, fontsize=9)
            else:
                ax_c.set_title(f"{status} fc={fc:.1f}x", color=status_color, fontsize=9)
            ax_c.axis('off')

    plt.suptitle(fname, color='white', fontsize=14, fontweight='bold', y=0.995)
    return fig


def _thicken_boundary(boundary_mask, radius=1):
    """Dilate a boundary mask to make outlines thicker."""
    from skimage.morphology import binary_dilation, disk
    if radius <= 0:
        return boundary_mask
    return binary_dilation(boundary_mask, disk(radius))


def _alpha_blend_outline(rgb_image, boundary_mask, color_rgb, alpha=0.7):
    """Alpha-blend a colored outline onto an RGB image (in-place)."""
    for c in range(3):
        rgb_image[:, :, c] = np.where(
            boundary_mask,
            alpha * color_rgb[c] + (1 - alpha) * rgb_image[:, :, c],
            rgb_image[:, :, c],
        )


def _make_dual_results_figure(red_image, green_image, labels, measurements,
                               summary, det_details, args):
    """Build a matplotlib figure for dual-channel colocalization results.

    Layout:
    - Top row: Magenta ch, Green ch, Clean Composite, Classification Overlay,
               Metrics+Legend (5 panels, last one narrower)
    - Middle rows: Zoomed nucleus panels (up to 6, 3 per row: red|green|composite)
    - Bottom row: Background diagnostic histograms (red ch, green ch)

    Outline strategy:
    - Overview panels: 2px thick, alpha-blended
    - Zoom panels: 3px thick, alpha-blended
    - Magenta panel: cyan outlines (max contrast against magenta)
    - Green panel: magenta outlines (max contrast against green)
    - Composite/Classification: category-colored outlines
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from skimage.segmentation import find_boundaries
    from skimage.measure import regionprops

    from mousebrain.plugin_2d.sliceatlas.core.colocalization import (
        normalize_for_display, DISPLAY_RED_FLOOR, DISPLAY_RED_MAX,
        DISPLAY_RED_GAMMA, DISPLAY_GRN_FLOOR, DISPLAY_GRN_MAX,
        DISPLAY_GRN_GAMMA,
    )

    r = normalize_for_display(red_image, DISPLAY_RED_MAX, DISPLAY_RED_FLOOR, DISPLAY_RED_GAMMA)
    g = normalize_for_display(green_image, DISPLAY_GRN_MAX, DISPLAY_GRN_FLOOR, DISPLAY_GRN_GAMMA)
    # Magenta + Green composite (white where both overlap)
    composite = np.stack([r, g, r], axis=-1)

    # Classification colors [R, G, B]
    cat_colors = {
        'dual':       [1.0, 1.0, 0.0],    # yellow
        'red_only':   [1.0, 0.27, 0.27],   # red
        'green_only': [0.27, 1.0, 0.27],   # green
        'neither':    [0.53, 0.53, 0.53],   # gray
    }

    # Build per-label classification lookup
    label_cat = {}
    label_fc_r = {}
    label_fc_g = {}
    ch1_name = summary.get('ch1_name', 'red')
    ch2_name = summary.get('ch2_name', 'green')
    if measurements is not None and len(measurements) > 0:
        for _, row in measurements.iterrows():
            lbl = int(row['label'])
            label_cat[lbl] = row.get('classification', 'neither')
            label_fc_r[lbl] = row.get(f'fold_change_{ch1_name}', 0)
            label_fc_g[lbl] = row.get(f'fold_change_{ch2_name}', 0)

    # Channel-specific measurement region boundaries
    from scipy import ndimage as _ndi
    from skimage.morphology import disk as _disk
    _selem_ch1 = _disk(args.ch1_soma_dilation) if args.ch1_soma_dilation > 0 else None
    _selem_ch2 = _disk(args.ch2_soma_dilation) if args.ch2_soma_dilation > 0 else None

    red_pos_boundary = np.zeros(labels.shape, dtype=bool)
    green_pos_boundary = np.zeros(labels.shape, dtype=bool)
    all_nuc_boundary = np.zeros(labels.shape, dtype=bool)
    dual_centroids = []

    for lbl, cat in label_cat.items():
        nuc_mask = labels == lbl
        nuc_bnd = find_boundaries(nuc_mask.astype(int), mode='outer')
        all_nuc_boundary |= nuc_bnd

        # Red+ cells: show their measurement region (nucleus or dilated)
        if cat in ('dual', 'red_only'):
            if _selem_ch1 is not None:
                ch1_roi = _ndi.binary_dilation(nuc_mask, structure=_selem_ch1)
                red_pos_boundary |= find_boundaries(ch1_roi.astype(int), mode='outer')
            else:
                red_pos_boundary |= nuc_bnd

        # Green+ cells: show their measurement region (soma or nucleus)
        if cat in ('dual', 'green_only'):
            if _selem_ch2 is not None:
                ch2_roi = _ndi.binary_dilation(nuc_mask, structure=_selem_ch2)
                green_pos_boundary |= find_boundaries(ch2_roi.astype(int), mode='outer')
            else:
                green_pos_boundary |= nuc_bnd

    # Dual centroids from measurements
    if measurements is not None and len(measurements) > 0:
        for _, row in measurements.iterrows():
            if row.get('classification') == 'dual':
                dual_centroids.append((row['centroid_y'], row['centroid_x']))

    red_pos_boundary_thick = _thicken_boundary(red_pos_boundary, radius=1)
    green_pos_boundary_thick = _thicken_boundary(green_pos_boundary, radius=1)
    all_nuc_boundary_thick = _thicken_boundary(all_nuc_boundary, radius=1)

    # Get nucleus regions for zoom panels
    props = regionprops(labels)
    props.sort(key=lambda p: p.area, reverse=True)
    n_zoom = min(6, len(props))
    zoom_pad = 40

    # ── Figure layout ──
    n_zoom_rows = max(1, (n_zoom + 2) // 3) if n_zoom > 0 else 0
    # Total height: top panels + zoom rows + histogram row
    fig_h = 6 + 4 * n_zoom_rows + 3.5
    fig = plt.figure(figsize=(28, fig_h))
    fig.patch.set_facecolor('#1a1a1a')
    fname = Path(args.file).stem

    h, w = labels.shape

    # Vertical layout fractions
    top_bottom = 0.55 if n_zoom > 0 else 0.38
    zoom_bottom = 0.28 if n_zoom > 0 else top_bottom
    hist_top = zoom_bottom - 0.03
    hist_bottom = 0.02

    # Top row: 5 panels (4 images + metrics), metrics narrower
    gs_top = fig.add_gridspec(1, 5, top=0.98, bottom=top_bottom,
                              left=0.02, right=0.98, wspace=0.04,
                              width_ratios=[1, 1, 1, 1, 0.7])

    OUTLINE_ALPHA = 0.7

    # Panel 1: Magenta channel — all detected nuclei (cyan outlines)
    ax1 = fig.add_subplot(gs_top[0, 0])
    red_rgb = np.stack([r, np.zeros_like(r), r], axis=-1).copy()
    _alpha_blend_outline(red_rgb, all_nuc_boundary_thick, [0, 1, 1], OUTLINE_ALPHA)
    ax1.imshow(np.clip(red_rgb, 0, 1))
    ax1.set_title(f"Magenta / mCherry: {labels.max()} nuclei",
                  color='#FF55FF', fontsize=10)
    ax1.axis('off')

    # Panel 2: Green channel — nucleus outlines + green soma outlines for green+ cells
    ax2 = fig.add_subplot(gs_top[0, 1])
    grn_rgb = np.stack([np.zeros_like(g), g, np.zeros_like(g)], axis=-1).copy()
    _alpha_blend_outline(grn_rgb, all_nuc_boundary_thick, [1, 0, 1], 0.5)
    _alpha_blend_outline(grn_rgb, green_pos_boundary_thick, [0.2, 1, 0.2], 0.85)
    ax2.imshow(np.clip(grn_rgb, 0, 1))
    ax2.set_title("Green / eYFP", color='#55FF55', fontsize=10)
    ax2.axis('off')

    # Panel 3: Clean composite (no outlines — just the merged image)
    ax3 = fig.add_subplot(gs_top[0, 2])
    ax3.imshow(np.clip(composite, 0, 1))
    ax3.set_title("Composite", color='white', fontsize=10)
    ax3.axis('off')

    # Panel 4: Classification — red nucleus outlines + green soma outlines + dual dots
    ax4 = fig.add_subplot(gs_top[0, 3])
    class_rgb = composite.copy()
    _alpha_blend_outline(class_rgb, red_pos_boundary_thick, [1, 0.33, 1], 0.85)
    _alpha_blend_outline(class_rgb, green_pos_boundary_thick, [0.33, 1, 0.33], 0.85)
    ax4.imshow(np.clip(class_rgb, 0, 1))
    # Dual-positive centroid markers
    for cy, cx in dual_centroids:
        ax4.plot(cx, cy, 'o', color='yellow', markersize=6,
                 markeredgecolor='black', markeredgewidth=0.5)
    # Zoom region boxes
    for i, prop in enumerate(props[:n_zoom]):
        cy, cx = prop.centroid
        y0 = max(0, int(cy) - zoom_pad)
        x0 = max(0, int(cx) - zoom_pad)
        sz = 2 * zoom_pad
        rect = Rectangle((x0, y0), sz, sz, linewidth=1.5,
                          edgecolor='cyan', facecolor='none', linestyle='--')
        ax4.add_patch(rect)
        ax4.text(x0 + 2, y0 - 2, str(i + 1), color='cyan', fontsize=8,
                 va='bottom', fontweight='bold')
    n_dual = summary.get('n_dual', 0)
    n_r = summary.get(f'n_{ch1_name}_only', 0)
    n_g = summary.get(f'n_{ch2_name}_only', 0)
    n_n = summary.get('n_neither', 0)
    ax4.set_title(
        f"Classification: Dual={n_dual}  Red={n_r}  Grn={n_g}  None={n_n}",
        color='white', fontsize=10,
    )
    ax4.axis('off')

    # Panel 5: Metrics text + legend
    ax5 = fig.add_subplot(gs_top[0, 4])
    ax5.set_facecolor('#1a1a1a')
    ax5.axis('off')

    total = summary.get('total_nuclei', 0)
    frac_r = summary.get(f'fraction_{ch1_name}', 0) * 100
    frac_g = summary.get(f'fraction_{ch2_name}', 0) * 100
    frac_d = summary.get('fraction_dual', 0) * 100
    bg_r = summary.get(f'background_{ch1_name}', 0)
    bg_g = summary.get(f'background_{ch2_name}', 0)

    thresh_str = f"{det_details['method']}: {det_details['threshold']:.0f}"
    if det_details.get('use_hysteresis'):
        thresh_str += f" (hyst low: {det_details['threshold_low']:.0f})"

    metrics_text = (
        f"File: {fname}\n\n"
        f"Detection\n"
        f"  Method: {thresh_str}\n"
        f"  Nuclei: {total}\n"
        f"  Raw: {det_details['raw_count']} -> "
        f"{det_details['filtered_count']}\n\n"
        f"Dual-Channel Classification\n"
        f"  Red+ (mCherry): "
        f"{summary.get(f'n_{ch1_name}_positive', 0)} "
        f"({frac_r:.1f}%)\n"
        f"  Green+ (eYFP):  "
        f"{summary.get(f'n_{ch2_name}_positive', 0)} "
        f"({frac_g:.1f}%)\n"
        f"  Dual+ (both):   {n_dual} ({frac_d:.1f}%)\n"
        f"  Red-only:       {n_r}\n"
        f"  Green-only:     {n_g}\n"
        f"  Neither:        {n_n}\n\n"
        f"Background\n"
        f"  Red bg:   {bg_r:.1f}\n"
        f"  Green bg: {bg_g:.1f}\n\n"
        f"Parameters\n"
        f"  Ch1 soma dil:  {args.ch1_soma_dilation}px\n"
        f"  Ch2 soma dil:  {args.ch2_soma_dilation}px\n"
        f"  Ch1 threshold: {args.ch1_threshold}x\n"
        f"  Ch2 threshold: {args.ch2_threshold}x\n"
    )
    ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes,
             color='white', fontsize=8, fontfamily='monospace',
             va='top', ha='left')

    # Color legend
    legend_items = [
        ('#FF55FF', 'Red+ nucleus'),
        ('#55FF55', 'Green+ soma'),
        ('#FFFF00', 'Dual+ (dot)'),
        ('cyan',    'Zoom region'),
    ]
    legend_y = 0.12
    for color, text in legend_items:
        ax5.plot([0.05], [legend_y], marker='s', markersize=7,
                 color=color, transform=ax5.transAxes)
        ax5.text(0.11, legend_y, text, transform=ax5.transAxes,
                 color='white', fontsize=7, va='center')
        legend_y -= 0.035

    # ── Middle rows: Zoomed nucleus panels ──
    if n_zoom > 0:
        n_cols = min(n_zoom, 3)
        n_z_rows = (n_zoom + n_cols - 1) // n_cols
        gs_bot = fig.add_gridspec(
            n_z_rows, n_cols * 3,
            top=top_bottom - 0.03, bottom=zoom_bottom,
            left=0.02, right=0.98,
            wspace=0.08, hspace=0.25,
        )

        for i, prop in enumerate(props[:n_zoom]):
            zrow = i // n_cols
            zcol = (i % n_cols) * 3

            cy, cx = int(prop.centroid[0]), int(prop.centroid[1])
            y0 = max(0, cy - zoom_pad)
            y1 = min(h, cy + zoom_pad)
            x0 = max(0, cx - zoom_pad)
            x1 = min(w, cx + zoom_pad)

            lbl = prop.label
            cat = label_cat.get(lbl, 'neither')
            fc_r = label_fc_r.get(lbl, 0)
            fc_g = label_fc_g.get(lbl, 0)

            r_crop = r[y0:y1, x0:x1]
            g_crop = g[y0:y1, x0:x1]
            comp_crop = composite[y0:y1, x0:x1]

            # Build per-channel measurement region contours
            lbl_crop = labels[y0:y1, x0:x1]
            nucleus_mask_crop = (lbl_crop == lbl)
            nucleus_contour = nucleus_mask_crop.astype(np.float64)

            # Ch1 (red) measurement region: nucleus or dilated soma
            if _selem_ch1 is not None:
                ch1_contour = _ndi.binary_dilation(
                    nucleus_mask_crop, structure=_selem_ch1
                ).astype(np.float64)
            else:
                ch1_contour = nucleus_contour

            # Ch2 (green) measurement region: nucleus or dilated soma
            if _selem_ch2 is not None:
                ch2_contour = _ndi.binary_dilation(
                    nucleus_mask_crop, structure=_selem_ch2
                ).astype(np.float64)
            else:
                ch2_contour = nucleus_contour

            # Category color for composite title
            cat_hex = {
                'dual': '#FFFF00', 'red_only': '#FF4444',
                'green_only': '#44FF44', 'neither': '#888888',
            }

            # Red channel zoom — red measurement region (cyan outline)
            ax_r = fig.add_subplot(gs_bot[zrow, zcol])
            red_crop = np.stack(
                [r_crop, np.zeros_like(r_crop), r_crop], axis=-1)
            ax_r.imshow(np.clip(red_crop, 0, 1))
            ax_r.contour(ch1_contour, levels=[0.5], linewidths=0.8,
                         colors=['cyan'], antialiased=True)
            ax_r.set_title(f"#{i+1} Red fc={fc_r:.1f}x",
                           color='white', fontsize=8)
            ax_r.axis('off')

            # Green channel zoom — green measurement region (magenta outline)
            ax_g = fig.add_subplot(gs_bot[zrow, zcol + 1])
            grn_crop = np.stack(
                [np.zeros_like(g_crop), g_crop, np.zeros_like(g_crop)],
                axis=-1)
            ax_g.imshow(np.clip(grn_crop, 0, 1))
            ax_g.contour(ch2_contour, levels=[0.5], linewidths=0.8,
                         colors=['magenta'], antialiased=True)
            ax_g.set_title(f"Green fc={fc_g:.1f}x",
                           color='white', fontsize=8)
            ax_g.axis('off')

            # Composite zoom — both measurement regions overlaid
            ax_c = fig.add_subplot(gs_bot[zrow, zcol + 2])
            ax_c.imshow(np.clip(comp_crop, 0, 1))
            # Red measurement region (solid magenta) — only if red+
            is_red_pos = cat in ('dual', 'red_only')
            is_green_pos = cat in ('dual', 'green_only')
            if is_red_pos:
                ax_c.contour(ch1_contour, levels=[0.5], linewidths=0.8,
                             colors=['#FF55FF'], antialiased=True,
                             linestyles='solid')
            # Green measurement region (dashed green) — only if green+
            if is_green_pos:
                ax_c.contour(ch2_contour, levels=[0.5], linewidths=0.8,
                             colors=['#55FF55'], antialiased=True,
                             linestyles='dashed')
            # Dual indicator: yellow dot at center
            if cat == 'dual':
                # Centroid relative to crop
                nuc_ys, nuc_xs = np.where(nucleus_mask_crop)
                if len(nuc_ys) > 0:
                    dot_y = nuc_ys.mean()
                    dot_x = nuc_xs.mean()
                    ax_c.plot(dot_x, dot_y, 'o', color='yellow',
                              markersize=5, markeredgecolor='black',
                              markeredgewidth=0.5)
            cat_labels = {
                'dual': 'DUAL', 'red_only': 'RED',
                'green_only': 'GRN', 'neither': '---',
            }
            cat_text_colors = {
                'dual': '#FFFF00', 'red_only': '#FF4444',
                'green_only': '#44FF44', 'neither': '#888888',
            }
            ax_c.set_title(
                cat_labels.get(cat, cat),
                color=cat_text_colors.get(cat, 'white'), fontsize=9,
            )
            ax_c.axis('off')

    # ── Bottom row: Background diagnostic histograms ──
    gs_hist = fig.add_gridspec(1, 2, top=hist_top, bottom=hist_bottom,
                               left=0.06, right=0.94, wspace=0.25)

    # Build tissue mask (exclude nuclei with generous dilation for green)
    from skimage.morphology import binary_dilation, disk
    nuclei_mask = labels > 0
    tissue_mask = red_image > 0  # nonzero = tissue (not background/empty)

    for ch_idx, (ax_pos, ch_img, ch_name_str, ch_color, bg_dil) in enumerate([
        (0, red_image, f"Red (561nm) — bg dilation={args.ch1_bg_dilation}px",
         '#FF55FF', args.ch1_bg_dilation),
        (1, green_image, f"Green (488nm) — bg dilation={args.ch2_bg_dilation}px",
         '#55FF55', args.ch2_bg_dilation),
    ]):
        ax_h = fig.add_subplot(gs_hist[0, ax_pos])
        ax_h.set_facecolor('#1a1a1a')

        # Tissue pixels outside dilated nuclei (what GMM sees)
        if bg_dil > 0:
            excl_mask = binary_dilation(nuclei_mask, disk(bg_dil))
        else:
            excl_mask = nuclei_mask
        bg_pixels = ch_img[tissue_mask & ~excl_mask].astype(np.float64)

        # Nucleus pixels (what is being measured)
        nuc_pixels = ch_img[nuclei_mask].astype(np.float64)

        if len(bg_pixels) > 0:
            # Clip to reasonable range for histogram
            p999 = np.percentile(np.concatenate([bg_pixels, nuc_pixels]), 99.9)
            bins = np.linspace(0, p999, 150)

            # Background pixel histogram
            ax_h.hist(bg_pixels, bins=bins, density=True, alpha=0.5,
                      color='#6688AA', label=f'Background tissue (n={len(bg_pixels):,})')

            # Nucleus pixel histogram
            if len(nuc_pixels) > 0:
                ax_h.hist(nuc_pixels, bins=bins, density=True, alpha=0.5,
                          color=ch_color,
                          label=f'Nucleus pixels (n={len(nuc_pixels):,})')

            # Overlay GMM fit if diagnostics available
            diag_key = f'bg_diagnostics_{["red", "green"][ch_idx]}'
            diag = summary.get(diag_key)
            if diag is not None:
                from scipy.stats import norm
                x = np.linspace(0, p999, 300)
                bg_mean = diag['background_mean']
                bg_std = diag['background_std']
                bg_wt = diag.get('background_weight', 1.0)

                # Background component
                y_bg = bg_wt * norm.pdf(x, bg_mean, bg_std)
                ax_h.plot(x, y_bg, '--', color='white', linewidth=1.5,
                          label=f'GMM bg: {bg_mean:.0f} +/- {bg_std:.0f}')

                # Signal component if 2-component
                if diag.get('n_components', 1) == 2:
                    sig_mean = diag['signal_bleed_mean']
                    sig_std = diag['signal_bleed_std']
                    sig_wt = diag.get('signal_bleed_weight', 0)
                    y_sig = sig_wt * norm.pdf(x, sig_mean, sig_std)
                    ax_h.plot(x, y_sig, '--', color='orange', linewidth=1.5,
                              label=f'GMM sig: {sig_mean:.0f} +/- {sig_std:.0f}')

                # Threshold line (background * fold_change)
                thresh_val = args.ch1_threshold if ch_idx == 0 else args.ch2_threshold
                thresh_line = bg_mean * thresh_val
                ax_h.axvline(thresh_line, color='#FF4444', linestyle=':',
                             linewidth=1.5,
                             label=f'Threshold: {thresh_val}x = {thresh_line:.0f}')

        ax_h.set_title(ch_name_str, color='white', fontsize=10)
        ax_h.set_xlabel('Intensity', color='white', fontsize=8)
        ax_h.set_ylabel('Density', color='white', fontsize=8)
        ax_h.tick_params(colors='white', labelsize=7)
        ax_h.legend(fontsize=7, facecolor='#333333', edgecolor='#555555',
                    labelcolor='white', loc='upper right')
        for spine in ax_h.spines.values():
            spine.set_color('#555555')

    plt.suptitle(f"{fname}  [DUAL]", color='white', fontsize=14,
                 fontweight='bold', y=0.995)
    return fig


def _save_results(output_dir, measurements, summary, labels, args, fig=None):
    """Save CSV, labels, figure, and summary to output directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.file).stem

    # Save measurements CSV
    csv_path = out / f"{stem}_colocalization.csv"
    measurements.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Save labels as TIFF
    import tifffile
    labels_path = out / f"{stem}_nuclei_labels.tif"
    tifffile.imwrite(str(labels_path), labels.astype(np.uint16))
    print(f"  Saved: {labels_path}")

    # Save figure
    if fig is not None:
        fig_path = out / f"{stem}_coloc_result.png"
        fig.savefig(str(fig_path), dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  Saved: {fig_path}")

    # Save summary
    summary_path = out / f"{stem}_summary.txt"
    with open(summary_path, 'w') as f:
        for k, v in summary.items():
            if k != 'coloc_metrics':
                f.write(f"{k}: {v}\n")
        if 'coloc_metrics' in summary:
            f.write("\nValidation Metrics:\n")
            for k, v in summary['coloc_metrics'].items():
                f.write(f"  {k}: {v}\n")
    print(f"  Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run detection + colocalization on a single image file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.nd2                          # Auto-detect everything
  %(prog)s image.nd2 --red-ch 1 --green-ch 0  # Specify channels
  %(prog)s image.nd2 --method percentile --percentile 97
  %(prog)s image.nd2 --min-area 5 --hysteresis-low 0.3
  %(prog)s image.nd2 --save results/          # Save CSV + labels
        """,
    )

    # Input
    parser.add_argument('file', help='Path to ND2 or TIFF file')

    # Channel selection
    parser.add_argument('--red-ch', type=int, default=None,
                        help='Nuclear (red) channel index (auto-detected if omitted)')
    parser.add_argument('--green-ch', type=int, default=None,
                        help='Signal (green) channel index (auto-detected if omitted)')

    # Detection parameters
    det = parser.add_argument_group('Detection')
    det.add_argument('--method', choices=['otsu', 'percentile', 'manual', 'zscore'],
                     default='zscore', help='Threshold method (default: zscore)')
    det.add_argument('--percentile', type=float, default=99.0,
                     help='Percentile for percentile method (default: 99)')
    det.add_argument('--manual-threshold', type=float, default=None,
                     help='Manual threshold value')
    det.add_argument('--min-area', type=int, default=10,
                     help='Minimum nucleus area in pixels (default: 10)')
    det.add_argument('--max-area', type=int, default=5000,
                     help='Maximum nucleus area in pixels (default: 5000)')
    det.add_argument('--opening-radius', type=int, default=0,
                     help='Morphological opening radius (default: 0)')
    det.add_argument('--closing-radius', type=int, default=0,
                     help='Morphological closing radius to bridge gaps (default: 0)')
    det.add_argument('--no-fill-holes', action='store_true',
                     help='Disable hole filling in binary mask')
    det.add_argument('--split-touching', action='store_true',
                     help='Watershed-split merged nuclei')
    det.add_argument('--split-footprint', type=int, default=10,
                     help='Footprint size for watershed peak detection (default: 10)')
    det.add_argument('--gaussian-sigma', type=float, default=1.0,
                     help='Gaussian blur sigma (default: 1.0)')
    det.add_argument('--no-hysteresis', action='store_true',
                     help='Disable hysteresis thresholding')
    det.add_argument('--hysteresis-low', type=float, default=0.5,
                     help='Hysteresis low fraction (default: 0.5)')
    det.add_argument('--min-solidity', type=float, default=0.0,
                     help='Min solidity filter (0=off, 0.7-0.8 typical for nuclei)')
    det.add_argument('--min-circularity', type=float, default=0.0,
                     help='Min circularity filter (0=off, 0.5 typical for nuclei)')

    # Colocalization parameters
    coloc = parser.add_argument_group('Colocalization')
    coloc.add_argument('--soma-dilation', type=int, default=0,
                       help='Soma dilation radius in pixels (default: 0 = nuclear)')
    coloc.add_argument('--bg-dilation', type=int, default=10,
                       help='Background estimation dilation (default: 10)')
    coloc.add_argument('--coloc-method',
                       choices=['adaptive', 'fold_change', 'area_fraction'],
                       default='adaptive',
                       help='Classification method (default: adaptive)')
    coloc.add_argument('--coloc-threshold', type=float, default=1.5,
                       help='Fold-change threshold for fallback (default: 1.5)')

    # Dual-channel colocalization
    dual = parser.add_argument_group('Dual Channel')
    dual.add_argument('--dual', action='store_true',
                      help='Enable dual-channel colocalization (both channels as signals)')
    dual.add_argument('--ch1-soma-dilation', type=int, default=0,
                      help='Soma dilation for channel 1/red (default: 0 = nuclear)')
    dual.add_argument('--ch2-soma-dilation', type=int, default=5,
                      help='Soma dilation for channel 2/green (default: 5 = cytoplasmic)')
    dual.add_argument('--ch1-threshold', type=float, default=2.0,
                      help='Fold-change threshold for channel 1 (default: 2.0)')
    dual.add_argument('--ch2-threshold', type=float, default=2.0,
                      help='Fold-change threshold for channel 2 (default: 2.0)')
    dual.add_argument('--ch1-bg-dilation', type=int, default=10,
                      help='Background exclusion dilation for ch1 (default: 10)')
    dual.add_argument('--ch2-bg-dilation', type=int, default=50,
                      help='Background exclusion dilation for ch2 (default: 50, generous for eYFP)')

    # Output
    parser.add_argument('--save', type=str, default=None,
                        help='Directory to save CSV, labels, and summary')
    parser.add_argument('--no-show', action='store_true',
                        help='Skip interactive display (useful with --save)')

    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    run_pipeline(args)


if __name__ == '__main__':
    main()
