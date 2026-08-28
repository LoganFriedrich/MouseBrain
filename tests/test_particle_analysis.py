"""
Standalone particle analysis test script.

Runs the ParticleAnalyzer on an ND2 file and shows results with matplotlib.
No napari, no full mousebrain stack - fast iteration for evaluating the tool.

Usage:
    python test_particle_analysis.py <nd2_file>
    python test_particle_analysis.py <nd2_file> --threshold 500 --min-area 20
    python test_particle_analysis.py <nd2_file> --det-channel 1 --meas-channel 0
"""

import argparse
import sys
import numpy as np


def load_nd2_channels(path):
    """Load channels from ND2 file. Returns list of 2D arrays + metadata."""
    try:
        import nd2
        f = nd2.ND2File(path)
        data = f.asarray()
        pixel_um = None
        try:
            pixel_um = f.metadata.channels[0].volume.axesCalibration[0]
        except Exception:
            pass
        f.close()
    except ImportError:
        from tifffile import imread
        data = imread(path)
        pixel_um = None

    # Handle shape: could be (C, Y, X) or (Y, X, C) or (Y, X)
    if data.ndim == 2:
        return [data], pixel_um
    elif data.ndim == 3:
        # Assume smallest dim is channels
        if data.shape[0] < data.shape[-1]:
            return [data[i] for i in range(data.shape[0])], pixel_um
        else:
            return [data[:, :, i] for i in range(data.shape[2])], pixel_um
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")


def main():
    parser = argparse.ArgumentParser(description="Standalone particle analysis test")
    parser.add_argument("nd2_file", help="Path to ND2 or TIFF file")
    parser.add_argument("--det-channel", type=int, default=1,
                        help="Detection channel index (default: 1 = red)")
    parser.add_argument("--meas-channel", type=int, default=0,
                        help="Measurement channel index (default: 0 = green)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Binarization threshold (default: Otsu auto)")
    parser.add_argument("--min-area", type=int, default=10,
                        help="Minimum particle area in pixels (default: 10)")
    parser.add_argument("--max-area", type=int, default=50000,
                        help="Maximum particle area in pixels (default: 50000)")
    parser.add_argument("--min-circ", type=float, default=0.0,
                        help="Minimum circularity 0-1 (default: 0)")
    parser.add_argument("--max-circ", type=float, default=1.0,
                        help="Maximum circularity 0-1 (default: 1)")
    parser.add_argument("--bg-method", default="percentile",
                        choices=["percentile", "median", "mean", "mode"],
                        help="Background estimation method (default: percentile)")
    parser.add_argument("--bg-percentile", type=float, default=50.0,
                        help="Background percentile (default: 50)")
    parser.add_argument("--save", default=None,
                        help="Save figure to this path instead of showing")
    args = parser.parse_args()

    # Load image
    print(f"Loading: {args.nd2_file}")
    channels, pixel_um = load_nd2_channels(args.nd2_file)
    print(f"  Channels: {len(channels)}, shape: {channels[0].shape}")
    if pixel_um:
        print(f"  Pixel size: {pixel_um:.4f} um/px")

    if args.det_channel >= len(channels) or args.meas_channel >= len(channels):
        print(f"ERROR: Only {len(channels)} channels available")
        sys.exit(1)

    det_img = channels[args.det_channel].astype(np.float64)
    meas_img = channels[args.meas_channel].astype(np.float64)
    print(f"  Detection channel {args.det_channel}: range [{det_img.min():.0f}, {det_img.max():.0f}]")
    print(f"  Measurement channel {args.meas_channel}: range [{meas_img.min():.0f}, {meas_img.max():.0f}]")

    # Auto threshold if not specified
    threshold = args.threshold
    if threshold is None:
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(det_img)
        print(f"  Auto threshold (Otsu): {threshold:.1f}")
    else:
        print(f"  Manual threshold: {threshold:.1f}")

    # Run analysis
    from mousebrain.plugin_2d.sliceatlas.core.particle_analysis import ParticleAnalyzer

    analyzer = ParticleAnalyzer()
    labels, results, measurements, summary = analyzer.run_full_analysis(
        detection_image=det_img,
        measurement_image=meas_img,
        threshold=threshold,
        min_area=args.min_area,
        max_area=args.max_area,
        min_circularity=args.min_circ,
        max_circularity=args.max_circ,
        background_method=args.bg_method,
        background_percentile=args.bg_percentile,
    )

    # Print results
    n = summary['n_particles']
    print(f"\n--- Results ---")
    print(f"  Particles found: {n}")
    print(f"  Mask coverage: {summary['mask_fraction']*100:.1f}%")
    if n > 0:
        print(f"  Background: {summary['background']:.1f}")
        print(f"  Mean particle intensity: {summary['mean_particle_intensity']:.1f}")
        print(f"  Mean above background: {summary['mean_above_background']:.1f}")
        print(f"  Mean SNR: {summary['mean_snr']:.2f}")
        print(f"  Mean area: {summary['mean_area']:.0f} px")
        if pixel_um:
            area_um2 = summary['mean_area'] * pixel_um**2
            print(f"  Mean area: {area_um2:.1f} um^2")
        print(f"\nPer-particle results:")
        print(results.to_string(index=False, max_rows=30))

    # Plot
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Particle Analysis: {args.nd2_file.split('/')[-1].split(chr(92))[-1]}", fontsize=12)

    # 1. Detection channel
    ax = axes[0, 0]
    vmax_det = np.percentile(det_img, 99.5)
    ax.imshow(det_img, cmap='gray', vmax=vmax_det)
    ax.set_title(f"Detection (ch{args.det_channel})")
    ax.axis('off')

    # 2. Binary mask
    ax = axes[0, 1]
    mask = det_img > threshold
    ax.imshow(mask, cmap='gray')
    ax.set_title(f"Mask (threshold={threshold:.0f})")
    ax.axis('off')

    # 3. Labeled particles on detection channel
    ax = axes[0, 2]
    ax.imshow(det_img, cmap='gray', vmax=vmax_det)
    if n > 0:
        # Overlay colored contours
        from scipy import ndimage
        for lbl in range(1, labels.max() + 1):
            single = (labels == lbl).astype(np.uint8)
            contour = ndimage.binary_dilation(single) & ~single.astype(bool)
            ax.contour(contour, colors=['cyan'], linewidths=0.5)
    ax.set_title(f"Particles: {n}")
    ax.axis('off')

    # 4. Measurement channel
    ax = axes[1, 0]
    vmax_meas = np.percentile(meas_img, 99.5)
    ax.imshow(meas_img, cmap='gray', vmax=vmax_meas)
    ax.set_title(f"Measurement (ch{args.meas_channel})")
    ax.axis('off')

    # 5. Measurement with particle overlays colored by intensity
    ax = axes[1, 1]
    ax.imshow(meas_img, cmap='gray', vmax=vmax_meas)
    if n > 0 and 'mean_intensity' in results.columns:
        for _, row in results.iterrows():
            lbl = int(row['label'])
            intensity = row.get('mean_above_background', row.get('mean_intensity', 0))
            color = 'lime' if intensity > 0 else 'red'
            single = (labels == lbl).astype(np.uint8)
            from scipy import ndimage as _ndi
            contour = _ndi.binary_dilation(single) & ~single.astype(bool)
            ax.contour(contour, colors=[color], linewidths=0.5)
    ax.set_title("Green=above bg, Red=below")
    ax.axis('off')

    # 6. Intensity distribution
    ax = axes[1, 2]
    if n > 0 and 'mean_intensity' in results.columns:
        bg = summary.get('background', 0)
        intensities = results['mean_intensity'].values
        ax.hist(intensities, bins=min(30, max(5, n // 3)), color='steelblue',
                edgecolor='white', alpha=0.8)
        ax.axvline(bg, color='red', linestyle='--', label=f'Background={bg:.0f}')
        ax.set_xlabel("Mean Intensity")
        ax.set_ylabel("Count")
        ax.legend()
    ax.set_title("Intensity Distribution")

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
