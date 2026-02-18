"""Round 7: FIXED channel order — ch0=green(signal), ch1=red(nuclear)."""
import numpy as np
from pathlib import Path
import nd2
from PIL import Image

nd2_path = Path(r"Y:\2_Connectome\Tissue\MouseBrain_Pipeline\2D_Slices\ENCR\ENCR_02_02_HD_Regions\Corrected\E02_02_S39_RN.nd2")
data = nd2.imread(str(nd2_path))

# Print channel info
f = nd2.ND2File(str(nd2_path))
ch_names = [ch.channel.name for ch in f.metadata.channels] if f.metadata and f.metadata.channels else []
print(f"Channel names: {ch_names}")
print(f"Shape: {data.shape}")
f.close()

# FIXED: ch0=green(signal/488), ch1=red(nuclear/561)
grn_raw = data[0].astype(np.float64)
red_raw = data[1].astype(np.float64)

print(f"Red  (ch1): mean={red_raw.mean():.1f} p99={np.percentile(red_raw, 99):.0f}")
print(f"Green(ch0): mean={grn_raw.mean():.1f} p99={np.percentile(grn_raw, 99):.0f}")

out_dir = Path(r"Y:\2_Connectome\Tissue\MouseBrain_Pipeline\2D_Slices\ENCR\ENCR_02_02_HD_Regions\ENCR_02_02")

def norm(img, floor, display_max, gamma):
    linear = np.clip((img - floor) / (display_max - floor), 0, 1)
    if gamma != 1.0:
        return np.power(linear, gamma)
    return linear

def save_comp(r, g, name, desc):
    comp = np.clip(np.stack([r, g, r], axis=-1), 0, 1)
    img = Image.fromarray((comp * 255).astype(np.uint8))
    path = out_dir / f"fullres_{name}.png"
    img.save(str(path))
    print(f"  {name}: {desc}")
    return path

r_p = {p: np.percentile(red_raw, p) for p in [50, 75, 90, 95, 99, 99.5, 99.9]}
g_p = {p: np.percentile(grn_raw, p) for p in [50, 75, 90, 95, 99, 99.5, 99.9]}
print(f"Red  pcts: p50={r_p[50]:.0f} p90={r_p[90]:.0f} p99={r_p[99]:.0f} p99.9={r_p[99.9]:.0f}")
print(f"Green pcts: p50={g_p[50]:.0f} p90={g_p[90]:.0f} p99={g_p[99]:.0f} p99.9={g_p[99.9]:.0f}")

paths = []

# Q2: Dim red nuclei, green provides tissue context
r = norm(red_raw, r_p[95], r_p[99.9], 1.0)
g = norm(grn_raw, g_p[50], g_p[99.9], 0.7)
paths.append(save_comp(r, g, "Q2",
    f"R[p95={r_p[95]:.0f}, p99.9] g=1.0 | G[p50={g_p[50]:.0f}, p99.9] g=0.7"))

# R2: Very dim red, green dominant
r = norm(red_raw, r_p[95], r_p[99.9], 2.0)
g = norm(grn_raw, g_p[75], g_p[99.9], 0.6)
paths.append(save_comp(r, g, "R2",
    f"R[p95, p99.9] g=2.0 | G[p75={g_p[75]:.0f}, p99.9] g=0.6"))

# S2: Minimal red, strong green tissue
r = norm(red_raw, r_p[99], r_p[99.9], 1.0)
g = norm(grn_raw, g_p[75], g_p[99.9], 0.5)
paths.append(save_comp(r, g, "S2",
    f"R[p99={r_p[99]:.0f}, p99.9] g=1.0 | G[p75, p99.9] g=0.5"))

# T2: Pure green only
r = np.zeros_like(red_raw)
g = norm(grn_raw, g_p[75], g_p[99.9], 0.6)
paths.append(save_comp(r, g, "T2",
    f"R=OFF | G[p75, p99.9] g=0.6"))

print(f"\nDone — {len(paths)} images at native resolution")
