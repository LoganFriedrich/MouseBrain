"""Full-resolution renders of best candidates M and N."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import nd2

nd2_path = Path(r"Y:\2_Connectome\Tissue\MouseBrain_Pipeline\2D_Slices\ENCR\ENCR_02_02_HD_Regions\Corrected\E02_02_S39_RN.nd2")
data = nd2.imread(str(nd2_path))
red_raw = data[0].astype(np.float64)
grn_raw = data[1].astype(np.float64)
h, w = red_raw.shape

def norm(img, floor, display_max, gamma):
    linear = np.clip((img - floor) / (display_max - floor), 0, 1)
    if gamma != 1.0:
        return np.power(linear, gamma)
    return linear

r_pcts = {p: np.percentile(red_raw, p) for p in [50, 75, 90, 99.5, 99.9]}
g_pcts = {p: np.percentile(grn_raw, p) for p in [50, 75, 90, 99.5, 99.9]}

out_dir = Path(r"Y:\2_Connectome\Tissue\MouseBrain_Pipeline\2D_Slices\ENCR\ENCR_02_02_HD_Regions\ENCR_02_02")

# Green: same for both (floor=p75, max=p99.9, gamma=0.7)
g_lo, g_hi, g_gamma = g_pcts[75], g_pcts[99.9], 0.7
g = norm(grn_raw, g_lo, g_hi, g_gamma)

configs = {
    "M": (r_pcts[75], r_pcts[99.9], 1.0,  "R[p75,p99.9] g=1.0"),
    "N": (r_pcts[75], r_pcts[99.5], 1.5,  "R[p75,p99.5] g=1.5"),
}

for name, (r_lo, r_hi, r_gamma, desc) in configs.items():
    r = norm(red_raw, r_lo, r_hi, r_gamma)
    comp = np.clip(np.stack([r, g, r], axis=-1), 0, 1)

    # Save composite as raw pixel array — 1:1 with source resolution
    # Use PIL for pixel-perfect output (no matplotlib resampling)
    from PIL import Image
    comp_uint8 = (comp * 255).astype(np.uint8)
    img = Image.fromarray(comp_uint8)
    path = out_dir / f"fullres_{name}_composite.png"
    img.save(str(path))
    print(f"Saved {name} composite: {path} ({img.size[0]}x{img.size[1]})")

    # Also save individual channels
    red_uint8 = (np.stack([r, np.zeros_like(r), r], axis=-1) * 255).astype(np.uint8)
    img_r = Image.fromarray(red_uint8)
    path_r = out_dir / f"fullres_{name}_red.png"
    img_r.save(str(path_r))

    grn_uint8 = (np.stack([np.zeros_like(g), g, np.zeros_like(g)], axis=-1) * 255).astype(np.uint8)
    img_g = Image.fromarray(grn_uint8)
    path_g = out_dir / f"fullres_{name}_green.png"
    img_g.save(str(path_g))

    print(f"  + red and green channels saved")

print("\nDone — all at native {0}x{1}".format(w, h))
