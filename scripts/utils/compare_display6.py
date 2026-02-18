"""Round 6: Green provides tissue context, red is dim nuclei markers only."""
import numpy as np
from pathlib import Path
import nd2
from PIL import Image

nd2_path = Path(r"Y:\2_Connectome\Tissue\MouseBrain_Pipeline\2D_Slices\ENCR\ENCR_02_02_HD_Regions\Corrected\E02_02_S39_RN.nd2")
data = nd2.imread(str(nd2_path))
red_raw = data[0].astype(np.float64)
grn_raw = data[1].astype(np.float64)

out_dir = Path(r"Y:\2_Connectome\Tissue\MouseBrain_Pipeline\2D_Slices\ENCR\ENCR_02_02_HD_Regions\ENCR_02_02")

def norm(img, floor, display_max, gamma):
    linear = np.clip((img - floor) / (display_max - floor), 0, 1)
    if gamma != 1.0:
        return np.power(linear, gamma)
    return linear

def save_composite(r, g, name, desc):
    comp = np.clip(np.stack([r, g, r], axis=-1), 0, 1)
    img = Image.fromarray((comp * 255).astype(np.uint8))
    path = out_dir / f"fullres_{name}.png"
    img.save(str(path))
    print(f"  {name}: {desc}")
    return path

r_p = {p: np.percentile(red_raw, p) for p in [50, 75, 90, 95, 99, 99.5, 99.9]}
g_p = {p: np.percentile(grn_raw, p) for p in [50, 75, 90, 95, 99, 99.5, 99.9]}
print(f"Red  p50={r_p[50]:.0f} p90={r_p[90]:.0f} p99={r_p[99]:.0f} p99.9={r_p[99.9]:.0f}")
print(f"Green p50={g_p[50]:.0f} p90={g_p[90]:.0f} p99={g_p[99]:.0f} p99.9={g_p[99.9]:.0f}")

# Philosophy:
#   - Green carries tissue context (dim autofluorescence) + bright signal
#   - Red is just faint nuclei markers — heavily suppressed
#   - Graininess fix: higher floor clips noise to black

paths = []

# Q: Very dim red, green carries tissue
#    Red: floor=p95 → only top 5% of red pixels visible at all
#    Green: floor=p50 → median clips noise, gamma=0.7 gently boosts tissue
r = norm(red_raw, r_p[95], r_p[99.9], 1.0)
g = norm(grn_raw, g_p[50], g_p[99.9], 0.7)
paths.append(save_composite(r, g, "Q",
    f"R[p95={r_p[95]:.0f}, p99.9] g=1.0 | G[p50={g_p[50]:.0f}, p99.9] g=0.7"))

# R: Even dimmer red, green slightly more aggressive floor
#    Red: floor=p95, gamma=2.0 → nuclei barely there
#    Green: floor=p75 → clips more noise, gamma=0.6 boosts tissue
r = norm(red_raw, r_p[95], r_p[99.9], 2.0)
g = norm(grn_raw, g_p[75], g_p[99.9], 0.6)
paths.append(save_composite(r, g, "R",
    f"R[p95, p99.9] g=2.0 | G[p75={g_p[75]:.0f}, p99.9] g=0.6"))

# S: Minimal red (almost green-only), smooth tissue
#    Red: floor=p99 → only brightest 1% of nuclei faintly visible
#    Green: floor=p75, gamma=0.5 → strong tissue boost, signal saturates
r = norm(red_raw, r_p[99], r_p[99.9], 1.0)
g = norm(grn_raw, g_p[75], g_p[99.9], 0.5)
paths.append(save_composite(r, g, "S",
    f"R[p99={r_p[99]:.0f}, p99.9] g=1.0 | G[p75, p99.9] g=0.5"))

# T: No red at all — pure green context + signal
#    Shows what green alone looks like as the tissue context provider
r = np.zeros_like(red_raw)
g = norm(grn_raw, g_p[75], g_p[99.9], 0.6)
paths.append(save_composite(r, g, "T",
    f"R=OFF | G[p75, p99.9] g=0.6 (green only)"))

print(f"\nSaved {len(paths)} full-resolution composites")
