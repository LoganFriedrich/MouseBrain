"""Compare display normalization — round 2, data-driven."""
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

# Print detailed percentiles to understand the distribution
for name, ch in [("Red", red_raw), ("Green", grn_raw)]:
    pcts = [50, 90, 95, 99, 99.5, 99.9, 99.99]
    vals = np.percentile(ch, pcts)
    print(f"{name}: min={ch.min():.0f} max={ch.max():.0f} mean={ch.mean():.1f}")
    for p, v in zip(pcts, vals):
        print(f"  p{p}: {v:.0f}")

def norm(img, floor, display_max, gamma):
    linear = np.clip((img - floor) / (display_max - floor), 0, 1)
    if gamma != 1.0:
        return np.power(linear, gamma)
    return linear

def composite(r, g):
    return np.clip(np.stack([r, g, r], axis=-1), 0, 1)

# Design philosophy:
#   - floor = subtle tissue autofluorescence visible (not pure black)
#   - max = positive cells are bright but not clipped
#   - gamma < 1 brightens dim tissue structure; gamma > 1 suppresses it

# ── E: Conservative — positive cells bright, bg just barely visible ──
rE = norm(red_raw, 20, 500, 0.7)
gE = norm(grn_raw, 25, 300, 0.6)

# ── F: Tissue-friendly — more bg structure, signal still pops ──
rF = norm(red_raw, 15, 400, 0.6)
gF = norm(grn_raw, 20, 200, 0.5)

# ── G: Aggressive tissue — max tissue structure, risk of noise ──
rG = norm(red_raw, 10, 300, 0.5)
gG = norm(grn_raw, 15, 150, 0.45)

# ── H: Percentile-based auto — p1 to p99.9 ──
r_lo, r_hi = np.percentile(red_raw, 1), np.percentile(red_raw, 99.9)
g_lo, g_hi = np.percentile(grn_raw, 1), np.percentile(grn_raw, 99.9)
rH = norm(red_raw, r_lo, r_hi, 0.6)
gH = norm(grn_raw, g_lo, g_hi, 0.6)

composites_list = [composite(rE, gE), composite(rF, gF),
                   composite(rG, gG), composite(rH, gH)]
greens_list = [gE, gF, gG, gH]
reds_list = [rE, rF, rG, rH]

titles = [
    f'E: Conservative\nR[20,500] g=0.7 | G[25,300] g=0.6',
    f'F: Tissue-friendly\nR[15,400] g=0.6 | G[20,200] g=0.5',
    f'G: Aggressive tissue\nR[10,300] g=0.5 | G[15,150] g=0.45',
    f'H: Auto percentile\nR[{r_lo:.0f},{r_hi:.0f}] g=0.6 | G[{g_lo:.0f},{g_hi:.0f}] g=0.6',
]

# 3 rows: composite, red-only, green-only
fig, axes = plt.subplots(3, 4, figsize=(32, 24), facecolor='#1a1a1a')

for col in range(4):
    # Row 0: composite
    axes[0, col].imshow(composites_list[col])
    axes[0, col].set_title(titles[col], color='white', fontsize=11, fontweight='bold')
    axes[0, col].axis('off')

    # Row 1: red only (magenta)
    r_disp = reds_list[col]
    red_rgb = np.stack([r_disp, np.zeros_like(r_disp), r_disp], axis=-1)
    axes[1, col].imshow(np.clip(red_rgb, 0, 1))
    axes[1, col].set_title('Red (magenta)', color='white', fontsize=10)
    axes[1, col].axis('off')

    # Row 2: green only
    g_disp = greens_list[col]
    grn_rgb = np.stack([np.zeros_like(g_disp), g_disp, np.zeros_like(g_disp)], axis=-1)
    axes[2, col].imshow(np.clip(grn_rgb, 0, 1))
    axes[2, col].set_title('Green only', color='white', fontsize=10)
    axes[2, col].axis('off')

for ax in axes.flat:
    ax.set_facecolor('black')

fig.suptitle('Display Normalization Comparison — Round 2 — E02_02_S39_RN',
             color='white', fontsize=16, fontweight='bold', y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])

out_path = Path(r"Y:\2_Connectome\Tissue\MouseBrain_Pipeline\2D_Slices\ENCR\ENCR_02_02_HD_Regions\ENCR_02_02\display_comparison2.png")
fig.savefig(str(out_path), dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.close(fig)
