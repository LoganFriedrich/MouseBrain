"""Compare two display normalization approaches side-by-side."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import nd2

# Load S39_RN (good test case — many positive cells + tissue structure)
nd2_path = Path(r"Y:\2_Connectome\Tissue\MouseBrain_Pipeline\2D_Slices\ENCR\ENCR_02_02_HD_Regions\Corrected\E02_02_S39_RN.nd2")
data = nd2.imread(str(nd2_path))
print(f"Shape: {data.shape}, dtype: {data.dtype}")

# Assume ch0=red, ch1=green (typical ENCR layout)
if data.ndim == 3 and data.shape[0] == 2:
    red_raw = data[0].astype(np.float64)
    grn_raw = data[1].astype(np.float64)
elif data.ndim == 3 and data.shape[2] == 2:
    red_raw = data[:, :, 0].astype(np.float64)
    grn_raw = data[:, :, 1].astype(np.float64)
else:
    raise ValueError(f"Unexpected shape: {data.shape}")

print(f"Red range: {red_raw.min():.0f} - {red_raw.max():.0f}, "
      f"mean={red_raw.mean():.1f}, p99={np.percentile(red_raw, 99):.0f}")
print(f"Green range: {grn_raw.min():.0f} - {grn_raw.max():.0f}, "
      f"mean={grn_raw.mean():.1f}, p99={np.percentile(grn_raw, 99):.0f}")

def norm(img, floor, display_max, gamma):
    linear = np.clip((img - floor) / (display_max - floor), 0, 1)
    if gamma != 1.0:
        return np.power(linear, gamma)
    return linear

# ── Approach A: run_coloc.py (current coloc_result.png) ──
rA = norm(red_raw, 25, 221, 0.8)
gA = norm(grn_raw, 32, 1333, 2.0)

# ── Approach B: visualization.py (QC figures) ──
rB = norm(red_raw, 0, 250, 0.7)
gB = norm(grn_raw, 200, 450, 0.7)

# ── Approach C: Hybrid — same floor/max as A, but moderate gamma ──
rC = norm(red_raw, 25, 221, 0.8)
gC = norm(grn_raw, 32, 1333, 0.5)  # gamma < 1 brightens midtones

# ── Approach D: Tight green range with subtle gamma ──
rD = norm(red_raw, 25, 250, 0.8)
gD = norm(grn_raw, 30, 600, 0.8)

# Build composites: magenta (R+B) + green
def composite(r, g):
    return np.stack([r, g, r], axis=-1)

compA = np.clip(composite(rA, gA), 0, 1)
compB = np.clip(composite(rB, gB), 0, 1)
compC = np.clip(composite(rC, gC), 0, 1)
compD = np.clip(composite(rD, gD), 0, 1)

# ── Figure: 2 rows x 4 cols (composite + green-only for each) ──
fig, axes = plt.subplots(2, 4, figsize=(32, 16), facecolor='#1a1a1a')

titles = [
    'A: run_coloc (current)\nR[25,221] g=0.8 | G[32,1333] g=2.0',
    'B: visualization.py\nR[0,250] g=0.7 | G[200,450] g=0.7',
    'C: Wide range, mild gamma\nR[25,221] g=0.8 | G[32,1333] g=0.5',
    'D: Medium range\nR[25,250] g=0.8 | G[30,600] g=0.8',
]
composites = [compA, compB, compC, compD]
greens = [gA, gB, gC, gD]

for col, (title, comp, g) in enumerate(zip(titles, composites, greens)):
    # Top row: composite
    ax = axes[0, col]
    ax.imshow(comp)
    ax.set_title(title, color='white', fontsize=11, fontweight='bold')
    ax.axis('off')

    # Bottom row: green channel only
    ax = axes[1, col]
    grn_rgb = np.stack([np.zeros_like(g), g, np.zeros_like(g)], axis=-1)
    ax.imshow(np.clip(grn_rgb, 0, 1))
    ax.set_title('Green channel only', color='white', fontsize=10)
    ax.axis('off')

for ax in axes.flat:
    ax.set_facecolor('black')

fig.suptitle('Display Normalization Comparison — E02_02_S39_RN',
             color='white', fontsize=16, fontweight='bold', y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])

out_path = Path(r"Y:\2_Connectome\Tissue\MouseBrain_Pipeline\2D_Slices\ENCR\ENCR_02_02_HD_Regions\ENCR_02_02\display_comparison.png")
fig.savefig(str(out_path), dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.close(fig)
