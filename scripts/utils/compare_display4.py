"""Compare display normalization — round 4: fix red oversaturation."""
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

r_pcts = {p: np.percentile(red_raw, p) for p in [50, 75, 90, 95, 99, 99.5, 99.9]}
g_pcts = {p: np.percentile(grn_raw, p) for p in [50, 75, 90, 95, 99, 99.5, 99.9]}
print("Red percentiles:", {k: f"{v:.0f}" for k, v in r_pcts.items()})
print("Green percentiles:", {k: f"{v:.0f}" for k, v in g_pcts.items()})

# Green: best from round 3 — floor=p75, max=p99.9, gamma=0.7
g_lo = g_pcts[75]
g_hi = g_pcts[99.9]
g_gamma = 0.7

# Red: the problem — need to gate so tissue bg is dim, nuclei pop
configs = [
    # (label, red_floor_pct, red_max_pct, red_gamma)
    ("M: R floor=p75, max=p99.9, g=1.0\n(linear, high floor)",
     75, 99.9, 1.0),
    ("N: R floor=p75, max=p99.5, g=1.5\n(high floor, suppress dim)",
     75, 99.5, 1.5),
    ("O: R floor=p90, max=p99.9, g=1.0\n(very high floor, linear)",
     90, 99.9, 1.0),
    ("P: R floor=p50, max=p99.9, g=2.0\n(median floor, strong suppress)",
     50, 99.9, 2.0),
]

fig, axes = plt.subplots(3, 4, figsize=(32, 24), facecolor='#1a1a1a')

for col, (title, r_floor_pct, r_max_pct, r_gamma) in enumerate(configs):
    r_lo = r_pcts[r_floor_pct]
    r_hi = np.percentile(red_raw, r_max_pct)

    r = norm(red_raw, r_lo, r_hi, r_gamma)
    g = norm(grn_raw, g_lo, g_hi, g_gamma)

    full_title = f"{title}\nR[{r_lo:.0f},{r_hi:.0f}] | G[{g_lo:.0f},{g_hi:.0f}] g={g_gamma}"

    comp = composite(r, g)
    axes[0, col].imshow(comp)
    axes[0, col].set_title(full_title, color='white', fontsize=10, fontweight='bold')
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

fig.suptitle('Display Normalization — Round 4: Red Channel Gating — E02_02_S39_RN',
             color='white', fontsize=16, fontweight='bold', y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])

out_path = Path(r"Y:\2_Connectome\Tissue\MouseBrain_Pipeline\2D_Slices\ENCR\ENCR_02_02_HD_Regions\ENCR_02_02\display_comparison4.png")
fig.savefig(str(out_path), dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.close(fig)
