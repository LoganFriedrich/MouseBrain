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

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend; we use os.startfile() to display PNGs

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

    # ── Compute physical size filters from metadata ──
    pixel_um = metadata.get('voxel_size_um', {}).get('x', None)
    if pixel_um and args.min_diameter_um is not None:
        # Convert diameter (µm) → area (pixels): A = π * (d / 2 / pixel_um)²
        min_area = int(np.pi * (args.min_diameter_um / 2 / pixel_um) ** 2)
        max_area = int(np.pi * (args.max_diameter_um / 2 / pixel_um) ** 2)
        print(f"\nPhysical size filter: {args.min_diameter_um}-{args.max_diameter_um} um "
              f"diameter -> {min_area}-{max_area} px area "
              f"(pixel size: {pixel_um:.2f} um/px)")
    else:
        min_area = args.min_area
        max_area = args.max_area

    # ── Detect nuclei ──
    print(f"Detecting nuclei (threshold, method={args.method})...")
    det_mod = _import_core('detection')
    detect_by_threshold = det_mod.detect_by_threshold
    t1 = time.time()

    labels, details = detect_by_threshold(
        red_image,
        method=args.method,
        percentile=args.percentile,
        manual_threshold=args.manual_threshold,
        min_area=min_area,
        max_area=max_area,
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
    if details.get('bg_median') is not None:
        print(f"  Z-score: bg_median={details['bg_median']:.1f}, "
              f"bg_std={details['bg_std']:.1f}, cutoff={details['z_cutoff']:.1f}sd")
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
    pixel_um = metadata.get('voxel_size_um', {}).get('x', None)
    if args.dual:
        fig = _make_dual_results_figure(
            red_image, green_image, labels, classified, summary,
            details, args, pixel_um=pixel_um,
        )
    else:
        fig = _make_results_figure(
            red_image, green_image, labels, classified, summary,
            details, coloc_metrics, args, pixel_um=pixel_um,
        )

    # Always save a PNG next to the input file for quick review
    png_path = Path(args.file).with_suffix('.coloc_result.png')
    fig.savefig(str(png_path), dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  Saved figure: {png_path}")

    # Open in system viewer for easy zoom/pan
    import os
    os.startfile(str(png_path))

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

    # Measure intensities — nuclear by default, cytoplasm ring if soma requested
    if args.soma_dilation > 0:
        measurements = analyzer.measure_cytoplasm_intensities(
            green_image, labels, expansion_px=args.soma_dilation,
        )
    else:
        measurements = analyzer.measure_nuclei_intensities(green_image, labels)

    # Classify
    classify_kw = {
        'method': args.coloc_method,
        'threshold': args.coloc_threshold,
    }
    if args.coloc_method in ('background_mean', 'local_snr', 'area_fraction'):
        classify_kw['signal_image'] = green_image
        classify_kw['nuclei_labels'] = labels
    if args.coloc_method == 'local_snr':
        classify_kw['local_snr_radius'] = getattr(args, 'local_snr_radius', 100)

    classified = analyzer.classify_positive_negative(
        measurements, background, **classify_kw,
    )

    summary = analyzer.get_summary_statistics(classified)

    # Print adaptive diagnostics if available
    if summary.get('adaptive_diagnostics'):
        ad = summary['adaptive_diagnostics']
        print(f"  Adaptive method: {ad['method_used']}")
        if ad['method_used'] == 'gmm_2component':
            gmm_col = ad.get('gmm_column', '?')
            print(f"    GMM column: {gmm_col}")
            print(f"    Negative: mean={ad['negative_mean']:.1f}, "
                  f"weight={ad['negative_weight']:.0%}")
            print(f"    Positive: mean={ad['positive_mean']:.1f}, "
                  f"weight={ad['positive_weight']:.0%}")
            print(f"    Separation: {ad['separation']:.2f}, "
                  f"threshold: {ad['adaptive_threshold']:.1f}")
            if 'ring_rescue' in ad:
                rr = ad['ring_rescue']
                print(f"    Ring rescue: {rr['n_rescued']} cells rescued "
                      f"(fc >= {rr['rescue_threshold_fc']:.2f}x, "
                      f"pos median fc={rr['pos_fc_median']:.2f}x)")
        elif ad['method_used'] == 'background_mean':
            print(f"    Background: mean={ad['background_mean']:.1f}, "
                  f"std={ad['background_std']:.1f}")
            print(f"    BG pixels: {ad['bg_pixels_used']} "
                  f"(excl radius: {ad['bg_excl_radius']}px)")
            n = ad['n_total']
            print(f"    Sigma binning ({n} nuclei):")
            print(f"      Above bg:     {ad['n_above_bg']:>4} "
                  f"({100*ad['n_above_bg']/n:.1f}%)")
            print(f"      Above 1 std:  {ad['n_above_1std']:>4} "
                  f"({100*ad['n_above_1std']/n:.1f}%)")
            print(f"      Above 1.5 std:{ad['n_above_1p5std']:>4} "
                  f"({100*ad['n_above_1p5std']/n:.1f}%)")
            print(f"      Above 2 std:  {ad['n_above_2std']:>4} "
                  f"({100*ad['n_above_2std']/n:.1f}%)")
            print(f"      Above 3 std:  {ad['n_above_3std']:>4} "
                  f"({100*ad['n_above_3std']/n:.1f}%)")
        elif ad['method_used'] == 'local_snr_otsu':
            print(f"    Otsu threshold: {ad['otsu_threshold']:.2f} sigma")
            print(f"    SNR range: median={ad['snr_median']:.2f}, "
                  f"max={ad['snr_max']:.2f}")
            print(f"    Above local bg: {ad['n_above_zero']} cells")
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

    ch1_method = getattr(args, 'ch1_method', 'fold_change')
    ch2_method = getattr(args, 'ch2_method', 'local_snr')
    use_ring = not getattr(args, 'no_cytoplasm_ring', False)

    print(f"\nRunning DUAL-CHANNEL colocalization...")
    print(f"  Ch1 (red):  method={ch1_method}, soma_dilation={args.ch1_soma_dilation}px, "
          f"threshold={args.ch1_threshold}, bg_dilation={args.ch1_bg_dilation}")
    print(f"  Ch2 (green): method={ch2_method}, soma_dilation={args.ch2_soma_dilation}px, "
          f"threshold={args.ch2_threshold}, bg_dilation={args.ch2_bg_dilation}")
    print(f"  Cytoplasm ring: {'ON (expand_labels)' if use_ring else 'OFF (legacy dilation)'}")

    coloc_mod = _import_core('colocalization')
    analyze_dual = coloc_mod.analyze_dual_colocalization
    t2 = time.time()

    classified, summary = analyze_dual(
        signal_image_1=red_image,
        signal_image_2=green_image,
        nuclei_labels=labels,
        threshold_method_1=ch1_method,
        threshold_value_1=args.ch1_threshold,
        cell_body_dilation_1=args.ch1_bg_dilation,
        soma_dilation_1=args.ch1_soma_dilation,
        threshold_method_2=ch2_method,
        threshold_value_2=args.ch2_threshold,
        cell_body_dilation_2=args.ch2_bg_dilation,
        soma_dilation_2=args.ch2_soma_dilation,
        use_cytoplasm_ring=use_ring,
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
                         summary, det_details, coloc_metrics, args,
                         pixel_um=None):
    """Build a matplotlib figure with detection + colocalization results.

    Layout:
    - Top row: Full image overview (red channel, composite, metrics text)
    - Bottom row: Zoomed panels around individual nuclei (up to 6),
      each showing red + green + composite with boundary outline
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from skimage.measure import regionprops

    # Normalize channels for display using auto-norm
    from mousebrain.plugin_2d.sliceatlas.core.colocalization import normalize_for_display_auto

    r = normalize_for_display_auto(red_image, channel='red')
    g = normalize_for_display_auto(green_image, channel='green')
    # Magenta + Green composite (white where both overlap)
    composite = np.stack([r, g, r], axis=-1)

    # Classify labels into tiers: positive / borderline / negative
    # Uses continuous local_snr score when available, otherwise binary
    positive_labels = set()
    borderline_labels = set()  # above background but below threshold
    negative_labels = set()
    label_fc = {}  # fold-change per label
    label_prob = {}  # posterior probability per label
    label_snr = {}  # local SNR score per label

    # Get Otsu threshold for tier boundaries
    otsu_thresh = None
    if summary and summary.get('adaptive_diagnostics'):
        ad = summary['adaptive_diagnostics']
        otsu_thresh = ad.get('otsu_threshold')

    if measurements is not None and len(measurements) > 0:
        has_snr = 'local_snr' in measurements.columns
        has_sigma = 'sigma_above_bg' in measurements.columns
        for _, row in measurements.iterrows():
            lbl = int(row['label'])
            label_fc[lbl] = row.get('fold_change', 0)
            if 'positive_probability' in row.index:
                label_prob[lbl] = row['positive_probability']
            if has_snr:
                snr = row['local_snr']
                label_snr[lbl] = snr
                if row.get('is_positive', False):
                    positive_labels.add(lbl)
                elif snr > 0:
                    borderline_labels.add(lbl)
                else:
                    negative_labels.add(lbl)
            elif has_sigma:
                # background_mean: tier by std devs above background
                # lime = >1 std above bg (strong), orange = 0-1 std (weak),
                # red = below bg
                sigma = row['sigma_above_bg']
                label_snr[lbl] = sigma  # reuse for display
                if sigma > 1.0:
                    positive_labels.add(lbl)
                elif sigma > 0:
                    borderline_labels.add(lbl)
                else:
                    negative_labels.add(lbl)
            else:
                if row.get('is_positive', False):
                    positive_labels.add(lbl)
                else:
                    negative_labels.add(lbl)

    # Per-category filled masks for smooth vector contour rendering
    pos_mask = np.zeros(labels.shape, dtype=np.float64)
    bord_mask = np.zeros(labels.shape, dtype=np.float64)
    neg_mask = np.zeros(labels.shape, dtype=np.float64)
    for lbl in positive_labels:
        pos_mask += (labels == lbl).astype(np.float64)
    for lbl in borderline_labels:
        bord_mask += (labels == lbl).astype(np.float64)
    for lbl in negative_labels:
        neg_mask += (labels == lbl).astype(np.float64)
    pos_mask = np.clip(pos_mask, 0, 1)
    bord_mask = np.clip(bord_mask, 0, 1)
    neg_mask = np.clip(neg_mask, 0, 1)
    # Smooth for vector-quality contours
    pos_mask_smooth = _smooth_mask(pos_mask) if pos_mask.any() else pos_mask
    bord_mask_smooth = _smooth_mask(bord_mask) if bord_mask.any() else bord_mask
    neg_mask_smooth = _smooth_mask(neg_mask) if neg_mask.any() else neg_mask
    all_nuc_smooth = _smooth_mask((labels > 0).astype(np.float64))

    # Halo ring for positive cells only
    if pos_mask.any():
        from skimage.morphology import binary_dilation, disk
        pos_halo_binary = binary_dilation(pos_mask > 0.5, disk(8))
        pos_halo_smooth = _smooth_mask(pos_halo_binary.astype(np.float64), sigma=1.5)
    else:
        pos_halo_smooth = pos_mask

    # Get nucleus regions for zoom panels
    props = regionprops(labels)
    zoom_pad = 60  # pixels of padding around each nucleus

    # Select spatially-spread zoom cells (no overlapping boxes)
    def _pick_spread_cells(candidates, n_pick, min_sep):
        """Pick up to n_pick cells with at least min_sep pixels between centroids."""
        picked = []
        for p in candidates:
            cy, cx = p.centroid
            too_close = False
            for q in picked:
                qy, qx = q.centroid
                if abs(cy - qy) < min_sep and abs(cx - qx) < min_sep:
                    too_close = True
                    break
            if not too_close:
                picked.append(p)
            if len(picked) >= n_pick:
                break
        return picked

    pos_props_sorted = sorted(
        [p for p in props if p.label in positive_labels], key=lambda p: -p.area)
    neg_props_sorted = sorted(
        [p for p in props if p.label in negative_labels], key=lambda p: -p.area)

    # Pick spread-out cells: up to 4 positive + 2 negative
    min_sep = zoom_pad * 2  # boxes must not overlap
    zoom_pos = _pick_spread_cells(pos_props_sorted, 4, min_sep)
    # For negative, also avoid overlap with already-picked positive
    all_picked = list(zoom_pos)
    for p in neg_props_sorted:
        cy, cx = p.centroid
        too_close = any(
            abs(cy - q.centroid[0]) < min_sep and abs(cx - q.centroid[1]) < min_sep
            for q in all_picked
        )
        if not too_close:
            all_picked.append(p)
        if len(all_picked) >= 6:
            break
    zoom_props_final = all_picked[:6]
    n_zoom = len(zoom_props_final)

    # ── Figure layout ──
    # Top row: 4 full-size images + 1 stats panel
    # Bottom row: up to 6 zoom panels (3 cols x 2 rows of triplets)
    n_zoom_rows = max(1, (n_zoom + 2) // 3)  # how many rows of zooms
    fig = plt.figure(figsize=(28, 6 + 4 * n_zoom_rows))
    fig.patch.set_facecolor('#1a1a1a')
    fname = Path(args.file).stem

    n_pos = summary['positive_cells']
    n_neg = summary['negative_cells']
    frac = summary['positive_fraction'] * 100

    # Top row: 4 images + 1 stats column
    gs_top = fig.add_gridspec(1, 5, top=0.93, bottom=0.55 if n_zoom > 0 else 0.02,
                              left=0.02, right=0.98, wspace=0.04,
                              width_ratios=[1, 1, 1, 1, 0.7])

    h, w = labels.shape

    # Panel 1: Magenta — all nuclei outlined (white contours)
    ax1 = fig.add_subplot(gs_top[0, 0])
    red_rgb = np.stack([r, np.zeros_like(r), r], axis=-1)
    ax1.imshow(np.clip(red_rgb, 0, 1))
    ax1.contour(all_nuc_smooth, levels=[0.5], linewidths=0.5,
                colors=['white'], alpha=0.7, antialiased=True)
    ax1.set_title(f"Nuclear ({labels.max()} nuclei)", color='white', fontsize=10)
    ax1.axis('off')

    # Panel 2: Green — only colocalized nuclei outlined
    ax2 = fig.add_subplot(gs_top[0, 1])
    grn_rgb = np.stack([np.zeros_like(g), g, np.zeros_like(g)], axis=-1)
    ax2.imshow(np.clip(grn_rgb, 0, 1))
    if pos_mask.any():
        ax2.contour(pos_mask_smooth, levels=[0.5], linewidths=0.6,
                    colors=['white'], alpha=0.8, antialiased=True)
    ax2.set_title(f"Signal ({n_pos} colocalized)", color='white', fontsize=10)
    ax2.axis('off')

    # Panel 3: Composite — three tiers: pos=lime, borderline=yellow, neg=dim red
    ax3 = fig.add_subplot(gs_top[0, 2])
    ax3.imshow(np.clip(composite, 0, 1))
    if neg_mask.any():
        ax3.contour(neg_mask_smooth, levels=[0.5], linewidths=0.3,
                    colors=['#ff4444'], alpha=0.3, antialiased=True)
    if bord_mask.any():
        ax3.contour(bord_mask_smooth, levels=[0.5], linewidths=0.5,
                    colors=['#ffaa00'], alpha=0.6, antialiased=True)
    if pos_mask.any():
        ax3.contour(pos_mask_smooth, levels=[0.5], linewidths=0.6,
                    colors=['lime'], alpha=0.7, antialiased=True)
        ax3.contour(pos_halo_smooth, levels=[0.5], linewidths=1.5,
                    colors=['yellow'], alpha=0.9, antialiased=True)
    n_bord_panel = len(borderline_labels)
    ax3.set_title(
        f"{len(positive_labels)} >1σ / {n_bord_panel} 0-1σ / "
        f"{len(negative_labels)} <bg",
        color='white', fontsize=10,
    )
    ax3.axis('off')

    # Panel 4: Composite + zoom boxes (same contours, non-overlapping boxes)
    ax4 = fig.add_subplot(gs_top[0, 3])
    ax4.imshow(np.clip(composite, 0, 1))
    if neg_mask.any():
        ax4.contour(neg_mask_smooth, levels=[0.5], linewidths=0.4,
                    colors=['#ff4444'], alpha=0.4, antialiased=True)
    if pos_mask.any():
        ax4.contour(pos_mask_smooth, levels=[0.5], linewidths=0.6,
                    colors=['lime'], alpha=0.7, antialiased=True)
        # Halo ring
        ax4.contour(pos_halo_smooth, levels=[0.5], linewidths=1.5,
                    colors=['yellow'], alpha=0.9, antialiased=True)
    # Non-overlapping zoom rectangles
    for i, prop in enumerate(zoom_props_final):
        cy, cx = prop.centroid
        y0 = max(0, int(cy) - zoom_pad)
        x0 = max(0, int(cx) - zoom_pad)
        sz = 2 * zoom_pad
        rect = Rectangle((x0, y0), sz, sz, linewidth=1,
                          edgecolor='yellow', facecolor='none', linestyle='--')
        ax4.add_patch(rect)
        ax4.text(x0 + 2, y0 - 2, str(i + 1), color='yellow', fontsize=8,
                 va='bottom', fontweight='bold')
    ax4.set_title("Zoom regions", color='white', fontsize=10)
    ax4.axis('off')

    # Panel 5: Stats — text + mini histogram
    # Use nested gridspec for histogram on top, text below
    gs_stats = gs_top[0, 4].subgridspec(2, 1, height_ratios=[1, 1.2], hspace=0.3)

    # Mini histogram — show sigma distribution for background_mean,
    # SNR for local_snr, fold-change as fallback
    ax_hist = fig.add_subplot(gs_stats[0])
    ax_hist.set_facecolor('#1a1a1a')
    has_sigma = measurements is not None and 'sigma_above_bg' in measurements.columns
    has_snr = measurements is not None and 'local_snr' in measurements.columns
    if has_sigma:
        sig_vals = measurements['sigma_above_bg'].dropna().values
        neg_sig = sig_vals[np.array([l in negative_labels for l in measurements['label']])]
        bord_sig = sig_vals[np.array([l in borderline_labels for l in measurements['label']])]
        pos_sig = sig_vals[np.array([l in positive_labels for l in measurements['label']])]
        bins = np.linspace(min(sig_vals.min(), -3), max(sig_vals.max(), 5), 35)
        if len(neg_sig) > 0:
            ax_hist.hist(neg_sig, bins=bins, color='#ff4444', alpha=0.5,
                         label=f'<bg ({len(neg_sig)})')
        if len(bord_sig) > 0:
            ax_hist.hist(bord_sig, bins=bins, color='#ffaa00', alpha=0.5,
                         label=f'0-1σ ({len(bord_sig)})')
        if len(pos_sig) > 0:
            ax_hist.hist(pos_sig, bins=bins, color='lime', alpha=0.6,
                         label=f'>1σ ({len(pos_sig)})')
        ax_hist.axvline(0, color='white', ls='--', lw=1, alpha=0.6,
                        label='bg mean')
        ax_hist.axvline(1.0, color='yellow', ls=':', lw=0.8, alpha=0.5,
                        label='1σ')
        ax_hist.set_xlabel('σ above background', color='#ccc', fontsize=7)
        ax_hist.tick_params(colors='#888', labelsize=6)
        ax_hist.legend(fontsize=5, loc='upper right', facecolor='#333',
                       edgecolor='#555', labelcolor='white')
        for spine in ax_hist.spines.values():
            spine.set_color('#555')
    elif has_snr:
        snr_vals = measurements['local_snr'].dropna().values
        pos_snr = snr_vals[measurements['is_positive'].values == True]
        bord_snr = snr_vals[np.array([l in borderline_labels for l in measurements['label']])]
        neg_snr = snr_vals[np.array([l in negative_labels for l in measurements['label']])]
        bins = np.linspace(min(snr_vals.min(), -2), max(snr_vals.max(), 5), 35)
        if len(neg_snr) > 0:
            ax_hist.hist(neg_snr, bins=bins, color='#ff4444', alpha=0.5,
                         label=f'Neg ({len(neg_snr)})')
        if len(bord_snr) > 0:
            ax_hist.hist(bord_snr, bins=bins, color='#ffaa00', alpha=0.5,
                         label=f'Bord ({len(bord_snr)})')
        if len(pos_snr) > 0:
            ax_hist.hist(pos_snr, bins=bins, color='lime', alpha=0.6,
                         label=f'Pos ({len(pos_snr)})')
        if otsu_thresh is not None:
            ax_hist.axvline(otsu_thresh, color='yellow', ls='--', lw=1, alpha=0.8,
                            label=f'Otsu={otsu_thresh:.1f}σ')
        ax_hist.axvline(0, color='white', ls=':', lw=0.5, alpha=0.4)
        ax_hist.set_xlabel('Local SNR (σ above bg)', color='#ccc', fontsize=7)
        ax_hist.tick_params(colors='#888', labelsize=6)
        ax_hist.legend(fontsize=5, loc='upper right', facecolor='#333',
                       edgecolor='#555', labelcolor='white')
        for spine in ax_hist.spines.values():
            spine.set_color('#555')
    elif measurements is not None and 'fold_change' in measurements.columns:
        fc_vals = measurements['fold_change'].dropna().values
        is_pos_col = measurements.get('is_positive', None)
        if is_pos_col is not None:
            pos_fc = measurements.loc[is_pos_col == True, 'fold_change'].values
            neg_fc = measurements.loc[is_pos_col != True, 'fold_change'].values
        else:
            pos_fc = np.array([])
            neg_fc = fc_vals
        bins = np.linspace(0, max(fc_vals.max(), 3), 30)
        if len(neg_fc) > 0:
            ax_hist.hist(neg_fc, bins=bins, color='#ff4444', alpha=0.6,
                         label=f'Neg ({len(neg_fc)})')
        if len(pos_fc) > 0:
            ax_hist.hist(pos_fc, bins=bins, color='lime', alpha=0.6,
                         label=f'Pos ({len(pos_fc)})')
        ad = summary.get('adaptive_diagnostics')
        if ad and ad.get('adaptive_threshold') and summary['background_used'] > 0:
            thresh_fc = ad['adaptive_threshold'] / summary['background_used']
            ax_hist.axvline(thresh_fc, color='yellow', ls='--', lw=1, alpha=0.8)
        ax_hist.set_xlabel('Fold change', color='#ccc', fontsize=7)
        ax_hist.tick_params(colors='#888', labelsize=6)
        ax_hist.legend(fontsize=6, loc='upper right', facecolor='#333',
                       edgecolor='#555', labelcolor='white')
        for spine in ax_hist.spines.values():
            spine.set_color('#555')
    ax_hist.set_title("Distribution", color='white', fontsize=9)

    # Metrics text
    ax_txt = fig.add_subplot(gs_stats[1])
    ax_txt.set_facecolor('#1a1a1a')
    ax_txt.axis('off')
    adapt_text = ""
    ad = summary.get('adaptive_diagnostics')
    if ad:
        if ad['method_used'] == 'background_mean':
            n_t = ad['n_total']
            adapt_text = (
                f"bg={ad['background_mean']:.1f} "
                f"std={ad['background_std']:.1f}\n"
                f"  >bg:   {ad['n_above_bg']:>3} "
                f"({100*ad['n_above_bg']/n_t:.0f}%)\n"
                f"  >1σ:   {ad['n_above_1std']:>3} "
                f"({100*ad['n_above_1std']/n_t:.0f}%)\n"
                f"  >1.5σ: {ad['n_above_1p5std']:>3} "
                f"({100*ad['n_above_1p5std']/n_t:.0f}%)\n"
                f"  >2σ:   {ad['n_above_2std']:>3} "
                f"({100*ad['n_above_2std']/n_t:.0f}%)\n"
                f"  >3σ:   {ad['n_above_3std']:>3} "
                f"({100*ad['n_above_3std']/n_t:.0f}%)\n"
            )
        elif ad['method_used'] == 'gmm_2component':
            adapt_text = (
                f"GMM: neg={ad['negative_mean']:.0f}"
                f"({ad['negative_weight']:.0%}) "
                f"pos={ad['positive_mean']:.0f}"
                f"({ad['positive_weight']:.0%})\n"
                f"Sep={ad['separation']:.1f} "
                f"thresh={ad['adaptive_threshold']:.0f}\n"
            )
        elif ad.get('reason'):
            adapt_text = f"Fallback: {ad['reason']}\n"
    measure_type = 'soma' if args.soma_dilation > 0 else 'nuclear'
    n_bord = len(borderline_labels)
    metrics_text = (
        f"{det_details['method']} detection\n"
        f"  {labels.max()} nuclei\n\n"
        f"{args.coloc_method}\n"
        f"  {measure_type} measurement\n"
        f"{adapt_text}\n"
        f"Pearson:  {coloc_metrics['pearson_r']:.3f}\n"
        f"M1(r>g):  {coloc_metrics['manders_m1']:.3f}\n"
        f"M2(g>r):  {coloc_metrics['manders_m2']:.3f}\n"
    )
    ax_txt.text(0.05, 0.95, metrics_text, transform=ax_txt.transAxes,
                color='white', fontsize=7, fontfamily='monospace',
                va='top', ha='left')

    # ── Bottom rows: Zoomed nucleus panels ──
    if n_zoom > 0:
        # zoom_props_final already selected above with spatial spread

        # Each zoom gets 3 sub-panels: red, green, composite
        n_cols = min(n_zoom, 3)
        n_rows = (n_zoom + n_cols - 1) // n_cols
        gs_bot = fig.add_gridspec(
            n_rows, n_cols * 3,
            top=0.48, bottom=0.02, left=0.02, right=0.98,
            wspace=0.08, hspace=0.25,
        )

        for i, prop in enumerate(zoom_props_final):
            row = i // n_cols
            col = (i % n_cols) * 3

            cy, cx = int(prop.centroid[0]), int(prop.centroid[1])
            y0 = max(0, cy - zoom_pad)
            y1 = min(h, cy + zoom_pad)
            x0 = max(0, cx - zoom_pad)
            x1 = min(w, cx + zoom_pad)

            lbl = prop.label
            is_pos = lbl in positive_labels

            # Crop regions
            r_crop = r[y0:y1, x0:x1]
            g_crop = g[y0:y1, x0:x1]
            comp_crop = composite[y0:y1, x0:x1]
            lbl_crop = labels[y0:y1, x0:x1]
            red_raw_crop = red_image[y0:y1, x0:x1]
            grn_raw_crop = green_image[y0:y1, x0:x1]

            # Find ALL labels in this crop and draw contours for each
            crop_labels = set(np.unique(lbl_crop)) - {0}
            n_crop_pos = 0
            n_crop_neg = 0

            # Voronoi territory map — each pixel assigned to nearest nucleus
            # Used to split adjacent green signals at the midline
            from skimage.segmentation import expand_labels
            territory_map = expand_labels(lbl_crop, distance=9999)

            # Build per-label contour masks — red-channel (nuclear) for red/composite,
            # green-channel (signal) for green panel
            cell_contours = []      # (nuc_mask, is_positive, label_id)
            green_contours = []     # (grn_mask, is_positive, label_id) — positive only
            for crop_lbl in crop_labels:
                lbl_is_pos = crop_lbl in positive_labels
                if lbl_is_pos:
                    n_crop_pos += 1
                else:
                    n_crop_neg += 1
                # Nuclear contour (from red channel) — for red + composite panels
                # Voronoi territory prevents adjacent nuclei from merging
                cmask = _intensity_contour_mask(
                    red_raw_crop, lbl_crop, crop_lbl, contour_dilation=4,
                    threshold_fraction=0.4,
                    territory_mask=(territory_map == crop_lbl),
                )
                cell_contours.append((cmask, lbl_is_pos, crop_lbl))
                # Green signal contour — for green panel (positive cells only)
                # Voronoi territory prevents adjacent cytoplasmic signals from merging
                # Higher threshold_fraction (0.5) for tighter contours
                if lbl_is_pos:
                    gmask = _intensity_contour_mask(
                        grn_raw_crop, lbl_crop, crop_lbl, contour_dilation=2,
                        threshold_fraction=0.5,
                        territory_mask=(territory_map == crop_lbl),
                    )
                    green_contours.append((gmask, crop_lbl))

            # Red channel zoom — white contours for all cells
            ax_r = fig.add_subplot(gs_bot[row, col])
            red_crop = np.stack([r_crop, np.zeros_like(r_crop), r_crop], axis=-1)
            ax_r.imshow(np.clip(red_crop, 0, 1))
            for cmask, lbl_pos, _ in cell_contours:
                ax_r.contour(cmask, levels=[0.5], linewidths=0.5,
                             colors=['white'], alpha=0.7, antialiased=True)
            ax_r.set_title(f"#{i+1} Red", color='white', fontsize=9)
            ax_r.axis('off')

            # Green channel zoom — contours trace actual green signal (not nuclei)
            ax_g = fig.add_subplot(gs_bot[row, col + 1])
            grn_crop_rgb = np.stack([np.zeros_like(g_crop), g_crop, np.zeros_like(g_crop)], axis=-1)
            ax_g.imshow(np.clip(grn_crop_rgb, 0, 1))
            for gmask, _ in green_contours:
                ax_g.contour(gmask, levels=[0.5], linewidths=0.5,
                             colors=['cyan'], alpha=0.7, antialiased=True)
            ax_g.set_title(f"Green", color='white', fontsize=9)
            ax_g.axis('off')

            # Composite zoom — 3-tier contours: pos=lime, borderline=orange, neg=red
            ax_c = fig.add_subplot(gs_bot[row, col + 2])
            ax_c.imshow(np.clip(comp_crop, 0, 1))
            for cmask, lbl_pos, crop_lbl_id in cell_contours:
                if lbl_pos:
                    cc, lw = 'lime', 0.6
                elif crop_lbl_id in borderline_labels:
                    cc, lw = '#ffaa00', 0.5
                else:
                    cc, lw = '#ff4444', 0.3
                ax_c.contour(cmask, levels=[0.5], linewidths=lw,
                             colors=[cc], alpha=0.7, antialiased=True)
            # Stats text
            stat_parts = []
            if n_crop_pos > 0:
                stat_parts.append(f"{n_crop_pos} pos")
            if n_crop_neg > 0:
                stat_parts.append(f"{n_crop_neg} neg")
            stat_str = ", ".join(stat_parts)
            # Primary cell title with sigma/SNR score
            sigma_val = label_snr.get(lbl)  # sigma_above_bg or local_snr
            if is_pos:
                primary_status = ">1σ"
                primary_color = 'lime'
            elif lbl in borderline_labels:
                primary_status = "0-1σ"
                primary_color = '#ffaa00'
            else:
                primary_status = "<bg"
                primary_color = '#ff4444'
            if sigma_val is not None:
                title_str = f"{primary_status} ({sigma_val:.1f}σ)"
            else:
                fc = label_fc.get(lbl, 0)
                title_str = f"{primary_status} fc={fc:.1f}x"
            ax_c.set_title(title_str, color=primary_color, fontsize=9)
            # Cell count annotation in bottom-left
            ax_c.text(0.03, 0.03, stat_str, transform=ax_c.transAxes,
                      color='white', fontsize=7, va='bottom',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                                alpha=0.6, edgecolor='none'))
            ax_c.axis('off')

    plt.suptitle(fname, color='white', fontsize=14, fontweight='bold', y=0.97)
    return fig


def _thicken_boundary(boundary_mask, radius=1):
    """Dilate a boundary mask to make outlines thicker."""
    from skimage.morphology import binary_dilation, disk
    if radius <= 0:
        return boundary_mask
    return binary_dilation(boundary_mask, disk(radius))


def _smooth_mask(mask, sigma=1.2):
    """Gaussian-smooth a binary mask so contour() produces smooth vector outlines.

    Without smoothing, contour() on a binary mask traces exact pixel edges,
    producing jagged staircase outlines. Smoothing the mask first makes the
    0.5 isoline follow a smooth interpolated curve — true vector rendering
    that looks like a fitted cell border rather than a pixelated ROI.

    Args:
        mask: Binary mask (bool or float, 0/1). Can be 2D.
        sigma: Gaussian sigma. 1.0-1.5 gives smooth outlines without
            distorting small cells. Larger = smoother but may merge
            nearby cells.
    Returns:
        Smoothed float mask suitable for contour(mask, levels=[0.5]).
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(mask.astype(np.float64), sigma=sigma)


def _intensity_contour_mask(channel_crop, lbl_crop, lbl, sigma=1.2,
                             threshold_fraction=0.3, contour_dilation=0,
                             territory_mask=None):
    """Create a contour mask from actual channel intensity instead of label shape.

    Instead of contouring the artificial circular label stamp, this traces
    the real cell boundary by thresholding the raw channel intensity and
    keeping the connected component at the detected nucleus centroid.

    Args:
        channel_crop: Raw intensity crop (not display-normalized).
        lbl_crop: Label array crop (same spatial extent).
        lbl: Which label ID to trace.
        sigma: Gaussian smoothing for vector-quality contour.
        threshold_fraction: Fraction between local background and peak
            intensity to place the contour (0.3 = 30% of the way up).
        contour_dilation: Pixels to dilate the mask outward before smoothing,
            so the contour frames the signal rather than overlapping it.
        territory_mask: Boolean mask of this cell's Voronoi territory
            (from expand_labels).  When provided, the intensity threshold
            is clipped to this territory so adjacent cells' signals split
            at the midline between their nuclei.
    Returns:
        Smoothed float mask suitable for contour(mask, levels=[0.5]).
    """
    from scipy.ndimage import label as ndi_label

    stamp = lbl_crop == lbl
    if not stamp.any():
        return _smooth_mask(stamp.astype(np.float64), sigma=sigma)

    # Centroid of the detected nucleus stamp
    ys, xs = np.where(stamp)
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))

    # Peak intensity — use territory if available (cytoplasmic signals
    # may peak outside the nuclear stamp), otherwise use the stamp
    if territory_mask is not None:
        peak_region = territory_mask
    else:
        peak_region = stamp
    peak_val = float(channel_crop[peak_region].max())

    # Local background: median of crop pixels outside all labels
    bg_pixels = channel_crop[lbl_crop == 0]
    local_bg = (float(np.median(bg_pixels)) if len(bg_pixels) > 0
                else float(np.percentile(channel_crop, 25)))

    if peak_val <= local_bg:
        return _smooth_mask(stamp.astype(np.float64), sigma=sigma)

    # Threshold at fraction between background and peak
    thresh = local_bg + threshold_fraction * (peak_val - local_bg)
    intensity_mask = channel_crop > thresh

    # Clip to cell's Voronoi territory — splits adjacent signals at midline
    if territory_mask is not None:
        intensity_mask = intensity_mask & territory_mask

    # Keep only the connected component containing the centroid
    labeled_cc, n_cc = ndi_label(intensity_mask)
    if n_cc == 0:
        return _smooth_mask(stamp.astype(np.float64), sigma=sigma)

    cy = min(max(cy, 0), labeled_cc.shape[0] - 1)
    cx = min(max(cx, 0), labeled_cc.shape[1] - 1)

    center_label = labeled_cc[cy, cx]
    if center_label == 0:
        # Centroid falls in background — fall back to stamp
        return _smooth_mask(stamp.astype(np.float64), sigma=sigma)

    cell_mask = (labeled_cc == center_label).astype(np.float64)

    # Dilate outward so contour frames the signal instead of overlapping it
    if contour_dilation > 0:
        from skimage.morphology import binary_dilation, disk
        cell_mask = binary_dilation(cell_mask > 0.5, disk(contour_dilation)).astype(np.float64)

    # Re-clip to territory after dilation so contours don't cross into neighbors
    if territory_mask is not None:
        cell_mask = cell_mask * territory_mask.astype(np.float64)

    return _smooth_mask(cell_mask, sigma=sigma)


def _add_scale_bar(ax, pixel_um, bar_um=None, location='lower right',
                   color='white', fontsize=7, height_fraction=0.015):
    """Draw a physical scale bar on a matplotlib axes.

    Args:
        ax: matplotlib axes (must already have an image displayed)
        pixel_um: microns per pixel
        bar_um: desired bar length in microns (auto-chosen if None)
        location: 'lower right' or 'lower left'
        color: bar color
        fontsize: label font size
        height_fraction: bar height as fraction of image height
    """
    if pixel_um is None or pixel_um <= 0:
        return
    # Get image extent from axes
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    img_w = abs(xlim[1] - xlim[0])
    img_h = abs(ylim[0] - ylim[1])  # ylim is inverted for images

    # Auto-choose bar length: ~15-20% of image width
    if bar_um is None:
        image_width_um = img_w * pixel_um
        # Pick a nice round number
        for candidate in [10, 20, 25, 50, 100, 200, 250, 500, 1000]:
            if candidate >= image_width_um * 0.12 and candidate <= image_width_um * 0.25:
                bar_um = candidate
                break
        if bar_um is None:
            bar_um = round(image_width_um * 0.18 / 10) * 10
            if bar_um < 1:
                bar_um = round(image_width_um * 0.18)
            if bar_um < 1:
                bar_um = 1

    bar_px = bar_um / pixel_um
    bar_h = img_h * height_fraction

    # Position
    margin_x = img_w * 0.03
    margin_y = img_h * 0.04
    if location == 'lower right':
        x0 = xlim[1] - margin_x - bar_px
        y0 = ylim[0] - margin_y  # ylim[0] is bottom for imshow
    else:
        x0 = xlim[0] + margin_x
        y0 = ylim[0] - margin_y

    from matplotlib.patches import FancyBboxPatch
    # Draw bar
    ax.plot([x0, x0 + bar_px], [y0, y0], color=color, linewidth=2, solid_capstyle='butt')
    # Label
    label = f"{bar_um:.0f} um" if bar_um >= 1 else f"{bar_um:.1f} um"
    ax.text(x0 + bar_px / 2, y0 - bar_h * 2, label,
            color=color, fontsize=fontsize, ha='center', va='top',
            fontweight='bold')


def _alpha_blend_outline(rgb_image, boundary_mask, color_rgb, alpha=0.7):
    """Alpha-blend a colored outline onto an RGB image (in-place)."""
    for c in range(3):
        rgb_image[:, :, c] = np.where(
            boundary_mask,
            alpha * color_rgb[c] + (1 - alpha) * rgb_image[:, :, c],
            rgb_image[:, :, c],
        )


def _make_dual_results_figure(red_image, green_image, labels, measurements,
                               summary, det_details, args, pixel_um=None):
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
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from skimage.measure import regionprops

    from mousebrain.plugin_2d.sliceatlas.core.colocalization import normalize_for_display_auto

    r = normalize_for_display_auto(red_image, channel='red')
    g = normalize_for_display_auto(green_image, channel='green')
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

    # Build filled masks for vector contour rendering (not pixelated boundaries)
    from scipy import ndimage as _ndi
    from skimage.morphology import disk as _disk
    _selem_ch1 = _disk(args.ch1_soma_dilation) if args.ch1_soma_dilation > 0 else None
    _selem_ch2 = _disk(args.ch2_soma_dilation) if args.ch2_soma_dilation > 0 else None

    # Filled masks — contour() will draw smooth vector outlines around these
    all_nuc_mask = (labels > 0).astype(np.float64)
    red_pos_mask = np.zeros(labels.shape, dtype=np.float64)
    green_pos_mask = np.zeros(labels.shape, dtype=np.float64)
    dual_centroids = []

    for lbl, cat in label_cat.items():
        nuc_mask = labels == lbl

        # Red+ measurement region mask
        if cat in ('dual', 'red_only'):
            if _selem_ch1 is not None:
                red_pos_mask += _ndi.binary_dilation(
                    nuc_mask, structure=_selem_ch1).astype(np.float64)
            else:
                red_pos_mask += nuc_mask.astype(np.float64)

        # Green+ measurement region mask
        if cat in ('dual', 'green_only'):
            if _selem_ch2 is not None:
                green_pos_mask += _ndi.binary_dilation(
                    nuc_mask, structure=_selem_ch2).astype(np.float64)
            else:
                green_pos_mask += nuc_mask.astype(np.float64)

    # Clip to binary (overlapping dilations can sum > 1)
    red_pos_mask = np.clip(red_pos_mask, 0, 1)
    green_pos_mask = np.clip(green_pos_mask, 0, 1)

    # Dual centroids
    if measurements is not None and len(measurements) > 0:
        for _, row in measurements.iterrows():
            if row.get('classification') == 'dual':
                dual_centroids.append((row['centroid_y'], row['centroid_x']))

    # Get nucleus regions for zoom panels — prioritize positive cells
    props = regionprops(labels)
    _cat_priority = {'dual': 0, 'red_only': 1, 'green_only': 2, 'neither': 3}
    props.sort(key=lambda p: (_cat_priority.get(label_cat.get(p.label, 'neither'), 3),
                               -p.area))
    n_zoom = min(6, len(props))
    zoom_pad = 60

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

    OVR_LW = 0.4   # overview contour linewidth (thin — don't obscure data)
    OVR_ALPHA = 0.3  # overview contour alpha (transparent — see through)

    # Smooth all masks for vector-quality contour rendering
    all_nuc_smooth = _smooth_mask(all_nuc_mask)
    red_pos_smooth = _smooth_mask(red_pos_mask) if red_pos_mask.any() else red_pos_mask
    green_pos_smooth = _smooth_mask(green_pos_mask) if green_pos_mask.any() else green_pos_mask

    # Panel 1: Magenta channel — all detected nuclei (cyan vector contours)
    ax1 = fig.add_subplot(gs_top[0, 0])
    red_rgb = np.stack([r, np.zeros_like(r), r], axis=-1)
    ax1.imshow(np.clip(red_rgb, 0, 1))
    ax1.contour(all_nuc_smooth, levels=[0.5], linewidths=OVR_LW,
                colors=['cyan'], alpha=OVR_ALPHA, antialiased=True)
    ax1.set_title(f"Magenta / mCherry: {labels.max()} nuclei",
                  color='#FF55FF', fontsize=10)
    ax1.axis('off')
    _add_scale_bar(ax1, pixel_um)

    # Panel 2: Green channel — nucleus + green-positive soma contours
    ax2 = fig.add_subplot(gs_top[0, 1])
    grn_rgb = np.stack([np.zeros_like(g), g, np.zeros_like(g)], axis=-1)
    ax2.imshow(np.clip(grn_rgb, 0, 1))
    ax2.contour(all_nuc_smooth, levels=[0.5], linewidths=0.3,
                colors=['magenta'], alpha=0.35, antialiased=True)
    if green_pos_mask.any():
        ax2.contour(green_pos_smooth, levels=[0.5], linewidths=OVR_LW,
                    colors=['#55FF55'], alpha=0.6, antialiased=True)
    ax2.set_title("Green / eYFP", color='#55FF55', fontsize=10)
    ax2.axis('off')
    _add_scale_bar(ax2, pixel_um)

    # Panel 3: Clean composite (no outlines)
    ax3 = fig.add_subplot(gs_top[0, 2])
    ax3.imshow(np.clip(composite, 0, 1))
    ax3.set_title("Composite", color='white', fontsize=10)
    ax3.axis('off')
    _add_scale_bar(ax3, pixel_um)

    # Panel 4: Classification — vector contours for each channel + dual dots
    ax4 = fig.add_subplot(gs_top[0, 3])
    ax4.imshow(np.clip(composite, 0, 1))
    if red_pos_mask.any():
        ax4.contour(red_pos_smooth, levels=[0.5], linewidths=OVR_LW,
                    colors=['#FF55FF'], alpha=0.6, antialiased=True)
    if green_pos_mask.any():
        ax4.contour(green_pos_smooth, levels=[0.5], linewidths=OVR_LW,
                    colors=['#55FF55'], alpha=0.6, antialiased=True,
                    linestyles='dashed')
    # Dual-positive arrows — point toward each colocalized cell from offset
    # shrinkB keeps the arrowhead AWAY from the cell so it doesn't obscure it
    for i_arrow, (cy, cx) in enumerate(dual_centroids):
        # Alternate arrow direction to reduce overlap
        angle = (i_arrow % 4) * 90 + 45  # 45, 135, 225, 315 degrees
        dx = 35 * np.cos(np.radians(angle))
        dy = 35 * np.sin(np.radians(angle))
        ax4.annotate(
            '', xy=(cx, cy), xytext=(cx + dx, cy + dy),
            arrowprops=dict(
                arrowstyle='->', color='yellow', lw=1.5,
                shrinkA=0, shrinkB=10,
            ),
        )
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
    _add_scale_bar(ax4, pixel_um)

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
        ('#FFFF00', 'Dual+ (arrow)'),
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

            # Intensity-based contours from raw channel data (actual cell shape)
            lbl_crop = labels[y0:y1, x0:x1]
            red_raw_crop = red_image[y0:y1, x0:x1]
            nucleus_contour = _intensity_contour_mask(
                red_raw_crop, lbl_crop, lbl)

            # Ch1 (red) measurement region: intensity-based nucleus or dilated
            if _selem_ch1 is not None:
                nuc_binary = nucleus_contour > 0.5
                ch1_contour = _smooth_mask(
                    _ndi.binary_dilation(
                        nuc_binary, structure=_selem_ch1,
                    ).astype(np.float64),
                    sigma=1.2,
                )
            else:
                ch1_contour = nucleus_contour

            # Ch2 (green) measurement region: dilated intensity-based nucleus
            if _selem_ch2 is not None:
                nuc_binary = nucleus_contour > 0.5
                ch2_contour = _smooth_mask(
                    _ndi.binary_dilation(
                        nuc_binary, structure=_selem_ch2,
                    ).astype(np.float64),
                    sigma=1.2,
                )
            else:
                ch2_contour = nucleus_contour

            # Category color for composite title
            cat_hex = {
                'dual': '#FFFF00', 'red_only': '#FF4444',
                'green_only': '#44FF44', 'neither': '#888888',
            }

            # Red channel zoom — both outlines (nucleus=cyan, soma=green dashed)
            ax_r = fig.add_subplot(gs_bot[zrow, zcol])
            red_crop = np.stack(
                [r_crop, np.zeros_like(r_crop), r_crop], axis=-1)
            ax_r.imshow(np.clip(red_crop, 0, 1))
            ax_r.contour(ch1_contour, levels=[0.5], linewidths=0.5,
                         colors=['cyan'], antialiased=True, alpha=0.6)
            ax_r.contour(ch2_contour, levels=[0.5], linewidths=0.4,
                         colors=['#55FF55'], antialiased=True,
                         linestyles='dashed', alpha=0.5)
            ax_r.set_title(f"#{i+1} Red fc={fc_r:.1f}x",
                           color='white', fontsize=8)
            ax_r.axis('off')
            _add_scale_bar(ax_r, pixel_um, fontsize=6)

            # Green channel zoom — both outlines (soma=magenta, nucleus=cyan dashed)
            ax_g = fig.add_subplot(gs_bot[zrow, zcol + 1])
            grn_crop = np.stack(
                [np.zeros_like(g_crop), g_crop, np.zeros_like(g_crop)],
                axis=-1)
            ax_g.imshow(np.clip(grn_crop, 0, 1))
            ax_g.contour(ch2_contour, levels=[0.5], linewidths=0.5,
                         colors=['magenta'], antialiased=True, alpha=0.6)
            ax_g.contour(ch1_contour, levels=[0.5], linewidths=0.4,
                         colors=['cyan'], antialiased=True,
                         linestyles='dashed', alpha=0.5)
            ax_g.set_title(f"Green fc={fc_g:.1f}x",
                           color='white', fontsize=8)
            ax_g.axis('off')
            _add_scale_bar(ax_g, pixel_um, fontsize=6)

            # Composite zoom — always show both measurement regions
            ax_c = fig.add_subplot(gs_bot[zrow, zcol + 2])
            ax_c.imshow(np.clip(comp_crop, 0, 1))
            # Red measurement region (solid magenta)
            ax_c.contour(ch1_contour, levels=[0.5], linewidths=0.5,
                         colors=['#FF55FF'], antialiased=True,
                         linestyles='solid', alpha=0.6)
            # Green measurement region (dashed green)
            ax_c.contour(ch2_contour, levels=[0.5], linewidths=0.5,
                         colors=['#55FF55'], antialiased=True,
                         linestyles='dashed', alpha=0.5)
            # Dual indicator: arrow pointing toward cell from corner
            # shrinkB keeps tip away from the cell so it doesn't obscure it
            if cat == 'dual':
                nuc_binary = nucleus_contour > 0.5
                nuc_ys, nuc_xs = np.where(nuc_binary)
                if len(nuc_ys) > 0:
                    tgt_y = nuc_ys.mean()
                    tgt_x = nuc_xs.mean()
                    crop_h, crop_w = nuc_binary.shape
                    ax_c.annotate(
                        '', xy=(tgt_x, tgt_y),
                        xytext=(crop_w * 0.9, crop_h * 0.1),
                        arrowprops=dict(
                            arrowstyle='->', color='yellow', lw=2,
                            shrinkA=0, shrinkB=8,
                        ),
                    )
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
            _add_scale_bar(ax_c, pixel_um, fontsize=6)

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
    det.add_argument('--min-diameter-um', type=float, default=6.0,
                     help='Minimum nucleus diameter in um (default: 6.0)')
    det.add_argument('--max-diameter-um', type=float, default=25.0,
                     help='Maximum nucleus diameter in µm (default: 25.0)')
    det.add_argument('--min-area', type=int, default=10,
                     help='Minimum nucleus area in pixels (fallback if no pixel size)')
    det.add_argument('--max-area', type=int, default=5000,
                     help='Maximum nucleus area in pixels (fallback if no pixel size)')
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
    det.add_argument('--min-circularity', type=float, default=0.3,
                     help='Min circularity filter (default: 0.3, nuclei are round-ish)')

    # Colocalization parameters
    coloc = parser.add_argument_group('Colocalization')
    coloc.add_argument('--soma-dilation', type=int, default=0,
                       help='Soma dilation radius in pixels (default: 0 = nuclear)')
    coloc.add_argument('--bg-dilation', type=int, default=10,
                       help='Background estimation dilation (default: 10)')
    coloc.add_argument('--coloc-method',
                       choices=['background_mean', 'zscore', 'adaptive',
                                'fold_change', 'area_fraction', 'local_snr'],
                       default='background_mean',
                       help='Classification method. '
                       'DEFAULT: background_mean — active method under '
                       'development. Computes tissue background mean (excluding '
                       'black areas + dilated somas) and classifies green ROI '
                       'above that mean as positive. All other methods are '
                       'retained for comparison during development: '
                       'local_snr, zscore, adaptive (GMM), fold_change, '
                       'area_fraction.')
    coloc.add_argument('--coloc-threshold', type=float, default=2.0,
                       help='Minimum threshold for local_snr (default: 2.0 sigma). '
                       'Otsu overrides this if it finds a higher natural break.')
    coloc.add_argument('--local-snr-radius', type=int, default=100,
                       help='Radius for local SNR background estimation '
                       '(default: 100px, used with --coloc-method local_snr)')

    # Dual-channel colocalization
    dual = parser.add_argument_group('Dual Channel')
    dual.add_argument('--dual', action='store_true',
                      help='Enable dual-channel colocalization (both channels as signals)')
    dual.add_argument('--ch1-soma-dilation', type=int, default=0,
                      help='Soma dilation for channel 1/red (default: 0 = nuclear)')
    dual.add_argument('--ch2-soma-dilation', type=int, default=8,
                      help='Soma dilation for channel 2/green (default: 8 = cytoplasmic)')
    dual.add_argument('--ch1-method', default='fold_change',
                      choices=['zscore', 'fold_change', 'adaptive', 'local_snr',
                               'area_fraction'],
                      help='Classification method for ch1 (default: fold_change)')
    dual.add_argument('--ch2-method', default='local_snr',
                      choices=['zscore', 'fold_change', 'adaptive', 'local_snr',
                               'area_fraction'],
                      help='Classification method for ch2 (default: local_snr, '
                      'best for dim cytoplasmic signals)')
    dual.add_argument('--ch1-threshold', type=float, default=2.0,
                      help='Threshold for channel 1 (default: 2.0x fold-change)')
    dual.add_argument('--ch2-threshold', type=float, default=3.0,
                      help='Threshold for channel 2 (default: 3.0 sigma for '
                      'local_snr, or fold-change for other methods)')
    dual.add_argument('--ch1-bg-dilation', type=int, default=10,
                      help='Background exclusion dilation for ch1 (default: 10)')
    dual.add_argument('--ch2-bg-dilation', type=int, default=50,
                      help='Background exclusion dilation for ch2 (default: 50, generous for eYFP)')
    dual.add_argument('--no-cytoplasm-ring', action='store_true',
                      help='Use legacy per-nucleus dilation instead of '
                      'expand_labels cytoplasm ring (not recommended)')

    # Output
    parser.add_argument('--save', type=str, default=None,
                        help='Directory to save CSV, labels, and summary')

    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    run_pipeline(args)


if __name__ == '__main__':
    main()
