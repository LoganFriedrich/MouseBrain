"""Diagnose classification: expose actual cutoff values and per-cell decisions.

Runs measurement + classification on E02_01_S13_DCN, then prints:
  1. The GMM background vs the internally-recomputed background_mean cutoff
  2. Per-cell intensity vs cutoff, highlighting borderline and changed calls

Usage:
    Y:\\LAB_ROOT\\envs\\MouseBrain\\python.exe test_validated_measurement.py
"""
import numpy as np
from pathlib import Path

ND2_PATH = Path(r"Y:\LAB_ROOT\Tissue\MouseBrain_Pipeline\2D_Slices\ENCR"
                r"\ENCR_02_01_HD_Regions\E02_01_S13_DCN.nd2")


def main():
    import nd2 as nd2lib
    from mousebrain.plugin_2d.sliceatlas.core.detection import detect_with_log_augmentation
    from mousebrain.plugin_2d.sliceatlas.core.colocalization import ColocalizationAnalyzer

    print("=" * 70)
    print("CLASSIFICATION DIAGNOSTIC: E02_01_S13_DCN")
    print("=" * 70)

    # Load
    print(f"\nLoading: {ND2_PATH.name}")
    data = nd2lib.imread(str(ND2_PATH))
    red = data[1]   # 561
    green = data[0]  # 488

    with nd2lib.ND2File(str(ND2_PATH)) as f:
        pixel_um = f.metadata.channels[0].volume.axesCalibration[0]
    print(f"  Pixel size: {pixel_um:.3f} um/px")
    print(f"  Image shape: {red.shape}")

    # Detect
    print("\nDetecting nuclei (threshold+LoG)...")
    labels, det = detect_with_log_augmentation(red, pixel_um=pixel_um)
    n_nuclei = det['filtered_count']
    print(f"  Found {n_nuclei} nuclei ({det['decision']}: "
          f"{det['n_threshold']} threshold + {det['n_log_new']} LoG)")

    # Detection diagnostics
    import math
    min_area = int(math.pi * (8.0 / 2 / pixel_um) ** 2)
    max_area = int(math.pi * (25.0 / 2 / pixel_um) ** 2)
    print(f"\n--- DETECTION DIAGNOSTICS ---")
    print(f"  Size filter: 8-25 um -> {min_area}-{max_area} px area")
    print(f"  Threshold (20% Otsu): {det.get('threshold', '?'):.1f}")
    print(f"  Otsu threshold: {det.get('otsu_threshold', '?'):.1f}")
    print(f"  LoG raw blobs found: {det.get('n_log_raw', '?')}")
    print(f"  LoG rejected (overlap): {det.get('n_log_rejected_overlap', '?')}")
    print(f"  LoG rejected (intensity): {det.get('n_log_rejected_intensity', '?')}")
    print(f"  LoG rejected (too small): {det.get('n_log_rejected_size', '?')}")
    print(f"  LoG accepted: {det.get('n_log_new', '?')}")
    print(f"  Removed by size (threshold): {det.get('removed_by_size', '?')}")
    print(f"  Trimmed (bottom 5% area): {det.get('n_trimmed_small', '?')}")
    print(f"  Removed by morphology (ecc/solidity): {det.get('removed_by_morphology', '?')}")
    print(f"  Intensity floor: {det.get('intensity_floor', '?')}")

    # Show detected nucleus sizes
    from skimage.measure import regionprops
    props = regionprops(labels)
    areas = [p.area for p in props]
    print(f"\n  Detected nucleus areas (px):")
    print(f"    min={min(areas)}, max={max(areas)}, "
          f"median={np.median(areas):.0f}, mean={np.mean(areas):.0f}")
    small = sum(1 for a in areas if a < 15)
    print(f"    nuclei with area < 15 px: {small}")

    # Morphology stats
    eccs = [p.eccentricity for p in props]
    sols = [p.solidity for p in props]
    print(f"\n  Eccentricity (0=circle, 1=line):")
    print(f"    min={min(eccs):.3f}, max={max(eccs):.3f}, "
          f"median={np.median(eccs):.3f}, mean={np.mean(eccs):.3f}")
    high_ecc = [(p.label, p.eccentricity, p.area, p.solidity) for p in props
                if p.eccentricity > 0.7]
    if high_ecc:
        high_ecc.sort(key=lambda t: -t[1])
        print(f"    High eccentricity nuclei (>0.7):")
        print(f"    {'Lbl':>4} {'ecc':>6} {'area':>5} {'solidity':>8}")
        for lbl, ecc, area, sol in high_ecc:
            print(f"    {lbl:>4} {ecc:>6.3f} {area:>5} {sol:>8.3f}")

    # Find bright spots in red that are NOT detected - these are likely
    # the missed nuclei the user sees
    from scipy.ndimage import gaussian_filter, maximum_filter
    from skimage.feature import blob_log
    red_smooth = gaussian_filter(red.astype(np.float32), sigma=1.0)
    # Find local maxima in red above the threshold
    thresh_20pct = det['threshold']
    local_max = maximum_filter(red_smooth, size=5) == red_smooth
    bright_spots = local_max & (red_smooth > thresh_20pct)
    bright_ys, bright_xs = np.where(bright_spots)
    # Check which ones are NOT in any label
    undetected = []
    for by, bx in zip(bright_ys, bright_xs):
        if labels[by, bx] == 0:
            # Not in any detected nucleus - check neighbors too
            has_neighbor = False
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    ny, nx = by + dy, bx + dx
                    if 0 <= ny < labels.shape[0] and 0 <= nx < labels.shape[1]:
                        if labels[ny, nx] > 0:
                            has_neighbor = True
                            break
                if has_neighbor:
                    break
            if not has_neighbor:
                undetected.append((by, bx, float(red[by, bx])))

    print(f"\n--- UNDETECTED BRIGHT SPOTS ---")
    print(f"  Bright spots above threshold ({thresh_20pct:.0f}) not in any label: "
          f"{len(undetected)}")
    if undetected:
        # Sort by intensity (brightest first)
        undetected.sort(key=lambda t: -t[2])
        print(f"  {'Y':>5} {'X':>5} {'RedInt':>7}")
        for by, bx, ri in undetected[:30]:
            print(f"  {by:>5} {bx:>5} {ri:>7.0f}")

    if n_nuclei == 0:
        print("No nuclei. Exiting.")
        return

    # Background estimation (two methods)
    soma_dil = 6
    analyzer = ColocalizationAnalyzer(background_method='gmm', background_percentile=10.0)
    bg_gmm = analyzer.estimate_background(green, labels, dilation_iterations=10)
    print(f"\nBackground (GMM, passed to classify): {bg_gmm:.2f}")

    # ── Measure with Voronoi ring (current method) ──
    meas = analyzer.measure_cytoplasm_intensities(green, labels, expansion_px=soma_dil)

    # ── Classify and capture diagnostics ──
    classified = analyzer.classify_positive_negative(
        meas, bg_gmm, method='background_mean',
        signal_image=green, nuclei_labels=labels, sigma_threshold=0,
    )
    diag = analyzer.adaptive_diagnostics

    print(f"\n--- BACKGROUND / CUTOFF ---")
    print(f"  GMM background (estimate_background):  {bg_gmm:.2f}")
    print(f"  bg_mean used by classify:              {diag['background_mean']:.2f}")
    print(f"  bg_std:                                {diag['background_std']:.2f}")
    print(f"  bg_source:                             {diag.get('bg_source', '?')}")
    print(f"  sigma_threshold:                       {diag['sigma_threshold']}")
    print(f"  ACTUAL pos_cutoff used:                {diag['positive_cutoff']:.2f}")

    # ── Per-cell details sorted by distance from cutoff ──
    cutoff = diag['positive_cutoff']
    print(f"\n--- PER-CELL: sorted by distance from cutoff ({cutoff:.1f}) ---")
    print(f"  Cells closest to the cutoff are the ones most likely misclassified.")
    print(f"\n  {'Lbl':>4} {'p75':>7} {'cutoff':>7} {'margin':>7} {'call':>4} "
          f"{'fold':>5} {'sigma':>6} {'nucArea':>7} {'cytoArea':>8}")
    print(f"  {'---':>4} {'---':>7} {'------':>7} {'------':>7} {'----':>4} "
          f"{'----':>5} {'-----':>6} {'-------':>7} {'--------':>8}")

    classified['margin'] = classified['soma_p75_intensity'] - cutoff
    sorted_df = classified.reindex(classified['margin'].abs().sort_values().index)

    for _, row in sorted_df.iterrows():
        lbl = int(row['label'])
        p75 = row['soma_p75_intensity']
        margin = row['margin']
        call = "POS" if row['is_positive'] else "NEG"
        fold = row['fold_change']
        sigma = row['sigma_above_bg']
        nuc_a = int(row['nucleus_area'])
        cyto_a = int(row['cyto_area'])
        flag = " <<<" if abs(margin) < 30 else ""
        print(f"  {lbl:>4} {p75:>7.1f} {cutoff:>7.1f} {margin:>+7.1f} {call:>4} "
              f"{fold:>5.2f} {sigma:>+6.1f} {nuc_a:>7} {cyto_a:>8}{flag}")

    # ── p75 vs mean: demonstrate noise bias ──
    print(f"\n--- P75 vs MEAN COMPARISON (noise bias diagnostic) ---")
    bg_std = diag['background_std']
    expected_p75_bias = 0.6745 * bg_std  # theoretical Q3 offset for Gaussian
    print(f"  Background mean:  {bg_gmm:.2f}")
    print(f"  Background std:   {bg_std:.2f}")
    print(f"  Expected p75 of pure noise: {bg_gmm:.1f} + 0.674*{bg_std:.1f} "
          f"= {bg_gmm + expected_p75_bias:.1f}")
    print(f"  So p75 inflates every cell by ~{expected_p75_bias:.1f} above true bg")
    print(f"\n  {'Lbl':>4} {'mean':>7} {'p75':>7} {'p75-mean':>8} {'mean>bg':>7} "
          f"{'p75>bg':>6} {'call_p75':>8} {'call_mean':>9}")
    n_flip = 0
    for _, row in classified.iterrows():
        lbl = int(row['label'])
        p75 = row['soma_p75_intensity']
        mean = row['soma_mean_intensity']
        diff = p75 - mean
        mean_above = mean - bg_gmm
        p75_above = p75 - bg_gmm
        call_p75 = "POS" if p75 > bg_gmm else "NEG"
        call_mean = "POS" if mean > bg_gmm else "NEG"
        flip = " <-- FLIP" if call_p75 != call_mean else ""
        if flip:
            n_flip += 1
        print(f"  {lbl:>4} {mean:>7.1f} {p75:>7.1f} {diff:>+8.1f} {mean_above:>+7.1f} "
              f"{p75_above:>+6.1f} {call_p75:>8} {call_mean:>9}{flip}")
    print(f"\n  Cells that flip POS->NEG if using mean instead of p75: {n_flip}")
    n_pos_mean = sum(1 for _, r in classified.iterrows()
                     if r['soma_mean_intensity'] > bg_gmm)
    n_pos_p75 = sum(1 for _, r in classified.iterrows()
                    if r['soma_p75_intensity'] > bg_gmm)
    print(f"  Positive by p75:  {n_pos_p75}/{len(classified)} ({n_pos_p75/len(classified)*100:.1f}%)")
    print(f"  Positive by mean: {n_pos_mean}/{len(classified)} ({n_pos_mean/len(classified)*100:.1f}%)")

    # ── Summary by sigma bins ──
    print(f"\n--- SIGMA DISTRIBUTION ---")
    print(f"  > 0 sigma (above bg):  {diag['n_above_bg']}")
    print(f"  > 1 sigma:             {diag['n_above_1std']}")
    print(f"  > 1.5 sigma:           {diag['n_above_1p5std']}")
    print(f"  > 2 sigma:             {diag['n_above_2std']}")
    print(f"  > 3 sigma:             {diag['n_above_3std']}")

    # ── METHOD COMPARISON: Voronoi vs Validated, p75 vs mean ──
    print(f"\n{'=' * 70}")
    print(f"METHOD COMPARISON (all use sigma=0, GMM bg={bg_gmm:.1f})")
    print(f"{'=' * 70}")

    # 1. Voronoi p75 (current method) - already computed above
    n_vor_p75 = int(classified['is_positive'].sum())

    # 2. Voronoi mean - reclassify using mean instead of p75
    n_vor_mean = int((classified['soma_mean_intensity'] > bg_gmm).sum())

    # 3. Validated method with neighbor exclusion
    import math
    consider_r = max(3, int(round(12.5 / pixel_um)))  # ~12.5 um
    excl_r = max(2, int(round(10.0 / pixel_um)))  # ~10 um exclusion
    print(f"  Validated: consideration_radius={consider_r}px "
          f"({consider_r*pixel_um:.1f}um), "
          f"exclusion_radius={excl_r}px ({excl_r*pixel_um:.1f}um)")

    meas_val = analyzer.measure_validated_intensities(
        green, labels,
        consideration_radius_px=consider_r,
        neighbor_exclusion_radius_px=excl_r,
        background=bg_gmm,
    )
    class_val = analyzer.classify_positive_negative(
        meas_val, bg_gmm, method='background_mean',
        signal_image=green, nuclei_labels=labels, sigma_threshold=0,
    )
    n_val_p75 = int(class_val['is_positive'].sum())
    n_val_mean = int((class_val['soma_mean_intensity'] > bg_gmm).sum())

    # 4. Validated with aggressive exclusion
    excl_r2 = max(3, int(round(15.0 / pixel_um)))  # ~15 um
    meas_val2 = analyzer.measure_validated_intensities(
        green, labels,
        consideration_radius_px=consider_r,
        neighbor_exclusion_radius_px=excl_r2,
        background=bg_gmm,
    )
    class_val2 = analyzer.classify_positive_negative(
        meas_val2, bg_gmm, method='background_mean',
        signal_image=green, nuclei_labels=labels, sigma_threshold=0,
    )
    n_val2_p75 = int(class_val2['is_positive'].sum())
    n_val2_mean = int((class_val2['soma_mean_intensity'] > bg_gmm).sum())

    print(f"\n  {'Method':<35} {'p75':>8} {'mean':>8}")
    print(f"  {'------':<35} {'---':>8} {'----':>8}")
    print(f"  {'Voronoi dil=6 (current)':<35} "
          f"{n_vor_p75:>3}/{len(classified):>3}   {n_vor_mean:>3}/{len(classified):>3}")
    print(f"  {'Validated excl={:.0f}um'.format(excl_r*pixel_um):<35} "
          f"{n_val_p75:>3}/{len(class_val):>3}   {n_val_mean:>3}/{len(class_val):>3}")
    print(f"  {'Validated excl={:.0f}um (aggressive)'.format(excl_r2*pixel_um):<35} "
          f"{n_val2_p75:>3}/{len(class_val2):>3}   {n_val2_mean:>3}/{len(class_val2):>3}")

    # Show per-cell differences between methods
    print(f"\n--- CELLS THAT DIFFER: Voronoi-p75 vs Validated-mean ---")
    print(f"  (Voronoi p75 = current method, Validated mean = both fixes combined)")
    print(f"\n  {'Lbl':>4} {'vor_p75':>8} {'vor_mean':>8} {'val_mean':>8} "
          f"{'vor_call':>8} {'val_call':>8} {'clean_px':>8} {'sig_dist':>8}")
    for _, rv in classified.iterrows():
        lbl = int(rv['label'])
        rvl = class_val[class_val['label'] == lbl]
        if len(rvl) == 0:
            continue
        rvl = rvl.iloc[0]
        call_vor = "POS" if rv['is_positive'] else "NEG"
        call_val = "POS" if rvl['soma_mean_intensity'] > bg_gmm else "NEG"
        if call_vor != call_val:
            print(f"  {lbl:>4} {rv['soma_p75_intensity']:>8.1f} "
                  f"{rv['soma_mean_intensity']:>8.1f} "
                  f"{rvl['soma_mean_intensity']:>8.1f} "
                  f"{call_vor:>8} {call_val:>8} "
                  f"{int(rvl['clean_area']):>8} "
                  f"{rvl['signal_distance']:>8.1f}"
                  f"  <-- {call_vor}->{call_val}")

    print()


if __name__ == '__main__':
    main()
