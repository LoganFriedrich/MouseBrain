"""Generate coloc result figure with ROI overlay and ROI-aware zoom picking.

Usage:
    python generate_coloc_roi_figure.py <nd2_path> <roi_json_path> <output_dir>
"""
import json, numpy as np, math, sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath

from mousebrain.plugin_2d.sliceatlas.core.detection import detect_with_log_augmentation
from mousebrain.plugin_2d.sliceatlas.core.colocalization import (
    ColocalizationAnalyzer, compute_colocalization_metrics
)
import mousebrain.plugin_2d.sliceatlas.cli.run_coloc as coloc_mod
from mousebrain.plugin_2d.sliceatlas.core import colocalization as coloc_core


def generate_figure(nd2_path, roi_path, out_dir, pixel_um=None):
    """Run coloc pipeline and generate figure with ROI overlay."""
    import nd2 as nd2lib

    nd2_path = Path(nd2_path)
    out_dir = Path(out_dir)
    stem = nd2_path.stem

    # Load ND2
    data = nd2lib.imread(str(nd2_path))
    # Handle Z-stacks: take max intensity projection
    if data.ndim == 4:  # (Z, C, H, W) or (C, Z, H, W)
        # Determine axis layout - try channel-first
        if data.shape[0] == 2:  # (C, Z, H, W)
            red = data[1].max(axis=0)
            green = data[0].max(axis=0)
        else:  # (Z, C, H, W)
            red = data[:, 1].max(axis=0)
            green = data[:, 0].max(axis=0)
    elif data.ndim == 3 and data.shape[0] == 2:
        red = data[1]   # 561
        green = data[0] # 488
    else:
        raise ValueError(f"Unexpected ND2 shape: {data.shape}")

    # Get pixel size from ND2 metadata if not provided
    if pixel_um is None:
        with nd2lib.ND2File(str(nd2_path)) as f:
            pixel_um = f.metadata.channels[0].volume.axesCalibration[0]

    # Detect using threshold + LoG augmentation with decision tree
    print(f"Detecting nuclei (threshold+LoG, 8-25 um, {pixel_um:.2f} um/px)...")
    labels, det_details = detect_with_log_augmentation(
        red, pixel_um=pixel_um,
    )
    n_nuclei = det_details['filtered_count']
    decision = det_details.get('decision', '?')
    n_thresh = det_details.get('n_threshold', 0)
    n_log = det_details.get('n_log_new', 0)
    print(f"Detected {n_nuclei} nuclei ({decision}: {n_thresh} threshold + {n_log} LoG)")

    if n_nuclei == 0:
        print("No nuclei found, skipping figure.")
        return None, {'n_nuclei': 0, 'n_positive': 0, 'rois': {}}

    # Colocalize with background_mean + Voronoi cytoplasm measurement
    # Voronoi territories prevent neighboring cells' green signal from overlapping
    soma_dil = 6  # expansion radius in pixels for cytoplasm ring
    analyzer = ColocalizationAnalyzer(background_method='gmm', background_percentile=10.0)
    background = analyzer.estimate_background(green, labels, dilation_iterations=10)
    tissue_mask = analyzer.estimate_tissue_mask(labels, 10)
    measurements = analyzer.measure_cytoplasm_intensities(
        green, labels, expansion_px=soma_dil)
    classified = analyzer.classify_positive_negative(
        measurements, background, method='background_mean',
        signal_image=green, nuclei_labels=labels, sigma_threshold=2,
    )
    summary = analyzer.get_summary_statistics(classified)
    coloc_metrics = compute_colocalization_metrics(
        red_image=red, green_image=green, nuclei_labels=labels,
        background_green=summary['background_used'],
        tissue_mask=tissue_mask, soma_dilation=soma_dil,
    )
    summary['coloc_metrics'] = coloc_metrics
    n_pos = summary['positive_cells']
    n_neg = summary['negative_cells']
    print(f"Positive: {n_pos}/{n_pos+n_neg} ({summary['positive_fraction']*100:.1f}%)")

    # Load ROIs
    with open(roi_path) as f:
        roi_data = json.load(f)

    roi_paths = []
    roi_names = []
    for roi in roi_data['rois']:
        verts = np.array(roi['vertices'])
        xy = verts[:, ::-1]  # (row,col) -> (col,row) for Path
        roi_paths.append(MplPath(xy))
        roi_names.append(roi['name'])

    # Classify cells by ROI membership
    from skimage.measure import regionprops
    props = regionprops(labels)
    pos_set = set(classified[classified['is_positive'] == True]['label'].astype(int))

    cells_in_roi = {i: [] for i in range(len(roi_paths))}
    cells_outside = []
    for p in props:
        cy, cx = p.centroid
        found = False
        for ri, rpath in enumerate(roi_paths):
            if rpath.contains_point([cx, cy]):
                cells_in_roi[ri].append(p)
                found = True
                break
        if not found:
            cells_outside.append(p)

    for ri, name in enumerate(roi_names):
        n_in = len(cells_in_roi[ri])
        n_p = sum(1 for p in cells_in_roi[ri] if p.label in pos_set)
        print(f"  ROI {name}: {n_in} cells ({n_p} positive)")
    print(f"  Outside: {len(cells_outside)} cells")

    # Edge distance for cell ranking
    from scipy.ndimage import distance_transform_edt
    noise_fl = max(np.percentile(red, 5), np.percentile(green, 5))
    tissue = (red > noise_fl) | (green > noise_fl)
    edge_dist = distance_transform_edt(tissue)

    def edist(p):
        cy, cx = p.centroid
        iy = max(0, min(int(round(cy)), edge_dist.shape[0] - 1))
        ix = max(0, min(int(round(cx)), edge_dist.shape[1] - 1))
        return edge_dist[iy, ix]

    # ROI-aware zoom picking: pick locations showing MULTIPLE cells per panel,
    # preferring views with a mix of POS and NEG classifications.
    zoom_pad = 60
    min_sep = zoom_pad * 2
    zoom_picks = []

    def no_overlap(candidate, picked):
        cy, cx = candidate.centroid
        for q in picked:
            qy, qx = q.centroid
            if abs(cy - qy) < min_sep and abs(cx - qx) < min_sep:
                return False
        return True

    def cells_in_crop(center_prop, all_props, pad=zoom_pad):
        """Count cells visible in a crop centered on center_prop."""
        cy, cx = center_prop.centroid
        n_pos, n_neg = 0, 0
        for p in all_props:
            py, px = p.centroid
            if abs(py - cy) < pad and abs(px - cx) < pad:
                if p.label in pos_set:
                    n_pos += 1
                else:
                    n_neg += 1
        return n_pos, n_neg

    def crop_score(prop, all_props):
        """Score a cell as zoom center: prefer crops showing many cells, mix of types."""
        n_pos, n_neg = cells_in_crop(prop, all_props)
        total = n_pos + n_neg
        # Best: both types visible. Bonus for more cells.
        has_mix = 1 if (n_pos > 0 and n_neg > 0) else 0
        return (has_mix, total, edist(prop), prop.area)

    # For each ROI: pick 2 best zoom locations (mix of POS/NEG preferred)
    for ri in range(len(roi_paths)):
        roi_cells = cells_in_roi[ri]
        sorted_cells = sorted(roi_cells, key=lambda p: crop_score(p, props), reverse=True)
        for c in sorted_cells:
            if sum(1 for x in zoom_picks if x in roi_cells) >= 2:
                break
            if no_overlap(c, zoom_picks):
                zoom_picks.append(c)

    # Outside cells: pick locations with good cell density and mix
    outside_scored = sorted(cells_outside, key=lambda p: crop_score(p, props), reverse=True)
    for c in outside_scored:
        if len(zoom_picks) >= 6:
            break
        if no_overlap(c, zoom_picks):
            zoom_picks.append(c)

    # Fill remaining
    for c in sorted(props, key=lambda p: crop_score(p, props), reverse=True):
        if len(zoom_picks) >= 6:
            break
        if c not in zoom_picks and no_overlap(c, zoom_picks):
            zoom_picks.append(c)

    print(f"Zoom picks: {len(zoom_picks)}")
    for zp in zoom_picks:
        cy, cx = zp.centroid
        status = "POS" if zp.label in pos_set else "NEG"
        loc = "outside"
        for ri, rp in enumerate(roi_paths):
            if rp.contains_point([cx, cy]):
                loc = f"in {roi_names[ri]}"
                break
        print(f"  Label {zp.label}: {status}, {loc}")

    # Generate figure with ROI-aware zoom override
    args = SimpleNamespace(
        file=str(nd2_path), coloc_method='background_mean',
        coloc_threshold=2.0, soma_dilation=soma_dil, sigma_threshold=2,
    )

    fig = coloc_mod._make_results_figure(
        red, green, labels, classified,
        summary, det_details, coloc_metrics, args,
        pixel_um=pixel_um,
        zoom_override=zoom_picks,
    )

    # No patches to restore

    # Draw ROIs on top-row overview panels (first 4 axes)
    all_axes = fig.get_axes()
    roi_colors = ['cyan', 'yellow']
    for ax_idx in range(min(4, len(all_axes))):
        ax = all_axes[ax_idx]
        for i, roi in enumerate(roi_data['rois']):
            raw_verts = np.array(roi['vertices'])
            xy_verts = raw_verts[:, ::-1]  # (row,col) -> (col,row)
            poly = MplPolygon(xy_verts, closed=True, fill=False,
                              edgecolor=roi_colors[i % 2], linewidth=2,
                              linestyle='--', alpha=0.85)
            ax.add_patch(poly)

    # ── ROI eYFP intensity comparison chart ──
    # For each location (In-ROI, Outside), show eYFP signal for:
    #   - All detected nuclei (red channel)
    #   - Colocalized subset (red + green)
    # Bars = mean +/- SD, with individual data points overlaid.
    roi_label_set = set()
    for ri in cells_in_roi:
        for p in cells_in_roi[ri]:
            roi_label_set.add(p.label)
    outside_label_set = set(p.label for p in cells_outside)

    combined_roi_name = " + ".join(roi_names) if len(roi_names) > 1 else (
        roi_names[0] if roi_names else "ROI")

    in_roi_all, in_roi_coloc = [], []
    out_all, out_coloc = [], []
    for _, row in classified.iterrows():
        lbl = int(row['label'])
        val = float(row['soma_p75_intensity'])
        is_pos = bool(row['is_positive'])
        if lbl in roi_label_set:
            in_roi_all.append(val)
            if is_pos:
                in_roi_coloc.append(val)
        elif lbl in outside_label_set:
            out_all.append(val)
            if is_pos:
                out_coloc.append(val)

    # Add chart as a new row below the existing figure
    fig_w, fig_h = fig.get_size_inches()
    chart_h = 3.0  # inches
    new_h = fig_h + chart_h
    fig.set_size_inches(fig_w, new_h)
    # Push existing content up
    frac = chart_h / new_h
    for ax in fig.get_axes():
        pos = ax.get_position()
        ax.set_position([
            pos.x0,
            pos.y0 * (1 - frac) + frac,
            pos.width,
            pos.height * (1 - frac),
        ])

    # Place chart centered in the bottom strip
    ax_bar = fig.add_axes([0.15, 0.03, 0.70, frac - 0.06])
    ax_bar.set_facecolor('#1a1a1a')

    # 4 bars: [ROI All, ROI Coloc+, Outside All, Outside Coloc+]
    groups = [
        (f"{combined_roi_name}\nAll nuclei", in_roi_all, '#ff6666'),
        (f"{combined_roi_name}\nColoc+", in_roi_coloc, '#00dddd'),
        ("Outside\nAll nuclei", out_all, '#cc5555'),
        ("Outside\nColoc+", out_coloc, '#009999'),
    ]

    bg_val = summary.get('background_used', 0)
    x_positions = [0, 1, 2.5, 3.5]
    all_vals = in_roi_all + out_all
    y_max = max(all_vals) * 1.15 if all_vals else 500

    rng = np.random.default_rng(42)
    for xi, (glabel, vals, color) in zip(x_positions, groups):
        n = len(vals)
        if n == 0:
            ax_bar.text(xi, y_max * 0.02, "n=0", ha='center', va='bottom',
                        color='#888', fontsize=9)
            continue

        arr = np.array(vals)
        mean_v = float(np.mean(arr))
        sd_v = float(np.std(arr, ddof=1)) if n > 1 else 0

        # Bar with SD error bar
        ax_bar.bar(xi, mean_v, 0.6, color=color, alpha=0.3,
                   edgecolor=color, linewidth=1.5)
        ax_bar.errorbar(xi, mean_v, yerr=sd_v, fmt='none',
                        ecolor='white', elinewidth=1.5, capsize=5,
                        capthick=1.5, zorder=6)

        # Individual data points (jittered)
        jitter = rng.uniform(-0.18, 0.18, n)
        ax_bar.scatter(xi + jitter, arr, s=20, color=color, alpha=0.8,
                       edgecolors='white', linewidths=0.3, zorder=5)

        # n count below bar
        ax_bar.text(xi, -y_max * 0.04, f"n={n}", ha='center', va='top',
                    color='white', fontsize=9, fontweight='bold')

    # eYFP background threshold line
    if bg_val > 0:
        ax_bar.axhline(bg_val, color='yellow', ls='--', lw=1.5, alpha=0.7,
                       label=f'eYFP background = {bg_val:.0f}')
        ax_bar.legend(fontsize=8, loc='upper right', facecolor='#333',
                      edgecolor='#555', labelcolor='white')

    ax_bar.set_xlim(-0.6, 4.3)
    ax_bar.set_ylim(-y_max * 0.07, y_max)
    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels([g[0] for g in groups], color='white', fontsize=9)
    ax_bar.set_ylabel('eYFP signal (a.u.)', color='white', fontsize=10)
    ax_bar.tick_params(colors='#aaa', labelsize=8)
    for spine in ax_bar.spines.values():
        spine.set_color('#555')
    ax_bar.set_title("eYFP signal: ROI vs Outside  (mean +/- SD, individual cells)",
                     color='white', fontsize=11, fontweight='bold')

    # Save
    fig_path = out_dir / f"{stem}_coloc_result.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # Register output to AnalysisRegistry
    try:
        from mousebrain.analysis_registry import AnalysisRegistry, get_approved_method
        registry = AnalysisRegistry(analysis_name="ENCR_ROI_Analysis")
        registry.register_output(
            sample=stem,
            category="roi_analysis",
            files={"figure": str(fig_path)},
            results={
                "n_nuclei": n_pos + n_neg,
                "n_positive": n_pos,
                "positive_fraction": summary['positive_fraction'],
                "background": summary.get('background_used', 0),
            },
            method_params=get_approved_method(),
            source_files={"nd2": str(nd2_path), "roi_json": str(roi_path)},
        )
        print(f"Registry: registered {stem} to ENCR_ROI_Analysis")
    except Exception as e:
        print(f"Registry warning: {e}")

    # Write ROI counts CSV for export
    import csv
    export_base = Path(r"Y:\2_Connectome\Databases\exports\ENCR_ROI_Analysis")
    # Determine subject/region from stem (e.g. E02_01_S13_DCN)
    parts = stem.split('_')
    if len(parts) >= 4:
        subj_dir = f"E{parts[0][1:]}_{parts[1]}"
        region_raw = parts[-1]
        base_region = region_raw.rstrip('0123456789')
        if base_region.endswith('v'):
            base_region = base_region[:-1]
        if base_region.endswith('Z'):
            base_region = base_region[:-1]
        export_dir = export_base / subj_dir / base_region
    else:
        export_dir = export_base
    export_dir.mkdir(parents=True, exist_ok=True)
    csv_path = export_dir / f"{stem}_roi_counts.csv"
    with open(csv_path, 'w', newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['roi', 'total', 'positive', 'negative', 'fraction'])
        total_all, pos_all = 0, 0
        for ri, name in enumerate(roi_names):
            roi_cells_ri = cells_in_roi[ri]
            n_in = len(roi_cells_ri)
            n_pos_in = sum(1 for p in roi_cells_ri if p.label in pos_set)
            n_neg_in = n_in - n_pos_in
            frac = n_pos_in / n_in if n_in > 0 else 0
            writer.writerow([name, n_in, n_pos_in, n_neg_in, frac])
            total_all += n_in
            pos_all += n_pos_in
        neg_all = total_all - pos_all
        frac_all = pos_all / total_all if total_all > 0 else 0
        writer.writerow(['TOTAL', total_all, pos_all, neg_all, frac_all])
    print(f"Export CSV: {csv_path}")

    # Build ROI stats dict for aggregation by batch script
    roi_stats = {
        'stem': stem,
        'n_nuclei': n_pos + n_neg,
        'n_positive': n_pos,
        'n_negative': n_neg,
        'positive_fraction': summary['positive_fraction'],
        'background': summary.get('background_used', 0),
        'rois': {},
    }
    for ri, name in enumerate(roi_names):
        roi_cells_ri = cells_in_roi[ri]
        n_in = len(roi_cells_ri)
        n_pos_in = sum(1 for p in roi_cells_ri if p.label in pos_set)
        roi_stats['rois'][name] = {
            'n_total': n_in,
            'n_positive': n_pos_in,
            'pct_positive': (n_pos_in / n_in * 100) if n_in > 0 else 0.0,
        }
    n_out = len(cells_outside)
    n_pos_out = sum(1 for p in cells_outside if p.label in pos_set)
    roi_stats['outside'] = {
        'n_total': n_out,
        'n_positive': n_pos_out,
        'pct_positive': (n_pos_out / n_out * 100) if n_out > 0 else 0.0,
    }

    return fig_path, roi_stats


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python generate_coloc_roi_figure.py <nd2> <roi_json> <output_dir>")
        sys.exit(1)
    generate_figure(sys.argv[1], sys.argv[2], sys.argv[3])
