"""Compare display normalization — round 3: auto-percentile variants."""
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

def norm(img, floor, display_max, gamma):
    linear = np.clip((img - floor) / (display_max - floor), 0, 1)
    if gamma != 1.0:
        return np.power(linear, gamma)
    return linear

def composite(r, g):
    return np.clip(np.stack([r, g, r], axis=-1), 0, 1)

# Red channel percentiles (same for all — red looked fine in H)
r_pcts = {p: np.percentile(red_raw, p) for p in [50, 75, 90, 95, 99, 99.5, 99.9]}
g_pcts = {p: np.percentile(grn_raw, p) for p in [50, 75, 90, 95, 99, 99.5, 99.9]}

print("Red percentiles:", {k: f"{v:.0f}" for k, v in r_pcts.items()})
print("Green percentiles:", {k: f"{v:.0f}" for k, v in g_pcts.items()})

# All use auto-percentile floors/maxes, vary how much background is suppressed
# Red: keep consistent — p1 floor, p99.9 max, gamma=0.7
r_lo = np.percentile(red_raw, 1)
r_hi = np.percentile(red_raw, 99.9)

configs = [
    # (label, green_floor_pct, green_max_pct, green_gamma)
    ("I: floor=p50, max=p99.9, g=0.8\n(median floor, subtle boost)", 50, 99.9, 0.8),
    ("J: floor=p75, max=p99.9, g=0.7\n(high floor, mod boost)", 75, 99.9, 0.7),
    ("K: floor=p90, max=p99.9, g=0.6\n(aggressive floor, boost midtones)", 90, 99.9, 0.6),
    ("L: floor=p50, max=p99.9, g=1.2\n(median floor, suppress dim)", 50, 99.9, 1.2),
]

fig, axes = plt.subplots(3, 4, figsize=(32, 24), facecolor='#1a1a1a')

for col, (title, g_floor_pct, g_max_pct, g_gamma) in enumerate(configs):
    g_lo = np.percentile(grn_raw, g_floor_pct)
    g_hi = np.percentile(grn_raw, g_max_pct)

    r = norm(red_raw, r_lo, r_hi, 0.7)
    g = norm(grn_raw, g_lo, g_hi, g_gamma)

    full_title = f"{title}\nG[{g_lo:.0f},{g_hi:.0f}]"

    comp = composite(r, g)
    axes[0, col].imshow(comp)
    axes[0, col].set_title(full_title, color='white', fontsize=11, fontweight='bold')
    axes[0, col].axis('off')

    red_rgb = np.stack([r, np.zeros_like(r), r], axis=-1)
    axes[1, col].imshow(np.clip(red_rgb, 0, 1))
    axes[1, col].set_title('Red (magenta)', color='white', fontsize=10)
    axes[1, col].axis('off')

    grn_rgb = np.stack([np.zeros_like(g), g, np.zeros_like(g)], axis=-1)
    axes[2, col].imshow(np.clip(grn_rgb, 0, 1))
    axes[2, col].set_title('Green only', color='white', fontsize=10)
    axes[2, col].axis('off')

for ax in axes.flat:
    ax.set_facecolor('black')

fig.suptitle('Display Normalization — Round 3: Auto-Percentile Variants — E02_02_S39_RN',
             color='white', fontsize=16, fontweight='bold', y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])

out_path = Path(r"Y:\2_Connectome\Tissue\MouseBrain_Pipeline\2D_Slices\ENCR\ENCR_02_02_HD_Regions\ENCR_02_02\display_comparison3.png")
fig.savefig(str(out_path), dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.close(fig)
