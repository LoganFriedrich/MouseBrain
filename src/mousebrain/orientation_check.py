"""Is a brain's declared orientation right? BrainGlobe's check, written as a picture.

WHY
---
Registration (brainreg) has to be told how the image stack lies: which way each
of its three axes runs ("iar", "psl", ...). Told wrongly, it registers anyway --
for over an hour -- and afterwards every cell lands in the wrong brain region
while every number still looks reasonable. BrainGlobe's own registration window
has a "Check orientation" button (brainreg.napari.register.check_orientation): it
turns the stack into the atlas's orientation AS DECLARED, averages it along each
of the three axes, and shows those three pictures under the same three pictures
of the atlas. If the declared orientation is right, each pair looks alike: same
way up, same left and right, front at the same end. That button needs a person
sitting at napari with the stack loaded. This module draws the same check as a
PNG, so the pipeline writes it for every brain before registering it, and a
person can look at one picture instead of opening a viewer. A re-stitched
delivery can come back lying differently from the first stitch.

HOW
---
* Reads every Nth slice and every Nth pixel of the registration input, so the
  result is at about the check atlas's resolution: plenty for a whole-brain
  outline, and a few GB of reading instead of tens.
* Maps that stack from the declared orientation into the atlas's orientation
  and resolution (brainglobe_space -- the same call BrainGlobe's button uses).
* Averages the stack and the atlas reference image along each axis and draws
  them in two rows, the atlas above, the brain below, each drawn at its real
  size in millimetres so a difference in shape or extent shows as such.
* Uses the 25 um version of the registration atlas when it is downloaded (same
  anatomy and orientation, a fraction of the memory); otherwise the atlas named.

Nothing here changes the stack or the registration. It only draws.

ASCII-only console output (Windows cp1252 consoles cannot print Unicode).
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

CHECK_RESOLUTION_UM = 25.0
CHECK_FILENAME = "QC_orientation_check.png"

# What each atlas section looks like to a person, by BrainGlobe's section name.
_VIEW_WORDS = {
    "frontal": "seen from the front (coronal)",
    "horizontal": "seen from above (horizontal)",
    "sagittal": "seen from the side (sagittal)",
}


def _slice_files(folder: Path):
    """The stack's slice files in order (Z0000.tif, Z0001.tif, ...)."""
    files = [p for p in Path(folder).iterdir()
             if p.suffix.lower() in (".tif", ".tiff") and not p.name.startswith(".")]

    def key(p):
        nums = re.findall(r"\d+", p.stem)
        return (int(nums[-1]) if nums else -1, p.name)
    return sorted(files, key=key)


def read_stack_subsampled(folder, voxel_um: Sequence[float],
                          target_um: float = CHECK_RESOLUTION_UM,
                          threads: int = 8) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """(stack, resolution in um as (z, y, x)) from a folder of slice images,
    keeping every Nth slice and pixel so the result is close to ``target_um``.
    ``voxel_um`` is the stack's (z, y, x) voxel size in micrometres."""
    import tifffile

    files = _slice_files(folder)
    if not files:
        raise FileNotFoundError("no .tif slices in %s" % folder)
    vz, vy, vx = (float(v) for v in voxel_um)
    z_step = max(1, int(round(target_um / vz)))
    y_step = max(1, int(round(target_um / vy)))
    x_step = max(1, int(round(target_um / vx)))
    chosen = files[::z_step]

    def one(p):
        return np.asarray(tifffile.imread(str(p)))[::y_step, ::x_step].astype(np.float32)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        planes = list(pool.map(one, chosen))
    return np.stack(planes), (vz * z_step, vy * y_step, vx * x_step)


def check_atlas_name(atlas_name: str) -> str:
    """The atlas to draw the check against: the 25 um sibling of ``atlas_name``
    when it is downloaded (same anatomy and orientation, far lighter), else
    ``atlas_name`` itself."""
    sibling = re.sub(r"_\d+um$", "_%dum" % int(CHECK_RESOLUTION_UM), atlas_name)
    if sibling != atlas_name:
        try:
            from brainglobe_atlasapi.list_atlases import get_downloaded_atlases
            if sibling in get_downloaded_atlases():
                return sibling
        except Exception:
            pass
    return atlas_name


def _show(img: np.ndarray) -> np.ndarray:
    """Contrast for display: stretched between the 1st and 99.5th percentile."""
    lo, hi = np.percentile(img, [1, 99.5])
    return np.clip((img - lo) / (hi - lo if hi > lo else 1.0), 0, 1)


def orientation_check(stack_folder, orientation: str, voxel_um: Sequence[float],
                      atlas_name: str, out_png, title: Optional[str] = None,
                      threads: int = 8) -> Dict:
    """Draw BrainGlobe's orientation check for one stack as a PNG at ``out_png``.

    ``orientation`` is the brainreg orientation declared for the stack (e.g.
    "iar"); ``voxel_um`` its (z, y, x) voxel size. Returns a summary: the PNG
    path, the atlas used, the declared orientation, the sampled resolution and
    the brain's extent in millimetres along each atlas axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from brainglobe_atlasapi import BrainGlobeAtlas
    from brainglobe_space import AnatomicalSpace

    stack, res = read_stack_subsampled(stack_folder, voxel_um, threads=threads)
    used = check_atlas_name(atlas_name)
    atlas = BrainGlobeAtlas(used, check_latest=False)
    source = AnatomicalSpace(orientation, shape=stack.shape, resolution=res)
    mapped = source.map_stack_to(atlas.space, stack)          # atlas orientation + resolution
    ref = np.asarray(atlas.reference, dtype=np.float32)
    ares = tuple(float(r) for r in atlas.resolution)

    sections = atlas.space.sections
    labels = atlas.space.axis_labels
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for i in range(3):
        keep = [k for k in range(3) if k != i]            # the two axes the picture shows
        for row, (vol, name) in enumerate(((ref, "Atlas (%s)" % used), (mapped, "This brain"))):
            proj = vol.mean(axis=i)
            h, w = proj.shape
            extent = [0, w * ares[keep[1]] / 1000.0, h * ares[keep[0]] / 1000.0, 0]
            ax = axes[row][i]
            ax.imshow(_show(proj), cmap="gray", extent=extent, interpolation="nearest")
            ax.set_title("%s, %s" % (name, _VIEW_WORDS.get(sections[i], sections[i])), fontsize=10)
            ax.set_ylabel("%s (mm)" % labels[i][0], fontsize=8)
            ax.set_xlabel("%s (mm)" % labels[i][1], fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle((title + "\n" if title else "")
                 + "Orientation check: declared '%s', shown in the atlas's orientation '%s'"
                 % (orientation, atlas.space.origin_string), fontsize=12)
    fig.text(0.5, 0.01,
             "If the declared orientation is right, each picture of the brain (bottom) lies the "
             "same way as the atlas above it: same way up, same left and right, front at the same "
             "end. A brain cut short by the person who prepared it only looks shorter.",
             ha="center", fontsize=9, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return {
        "png": str(out_png),
        "declared_orientation": orientation,
        "atlas": used,
        "atlas_orientation": atlas.space.origin_string,
        "sampled_resolution_um": [round(r, 3) for r in res],
        "brain_extent_mm": [round(mapped.shape[k] * ares[k] / 1000.0, 2) for k in range(3)],
        "atlas_extent_mm": [round(ref.shape[k] * ares[k] / 1000.0, 2) for k in range(3)],
    }


# ---------------------------------------------------------------------------
# One brain, as the pipeline registers it
# ---------------------------------------------------------------------------

# The registration step's atlas when none is named (scripts/pipeline/3_register_to_atlas.py).
DEFAULT_ATLAS = "allen_mouse_10um"
REGISTRY_ANALYSIS = "3D_Orientation_Check"
CHECK_VERSION = "1.0.0"


def brain_inputs(pipeline_dir, source: str = "auto") -> Dict:
    """What registration reads for this brain: the stack folder (registration
    channel of the best available input, exactly as 3_register_to_atlas picks
    it), its (z, xy, xy) voxel size and the declared orientation. Raises when
    the brain has no images."""
    from mousebrain.config import resolve_pipeline_input

    source_name, folder, metadata = resolve_pipeline_input(pipeline_dir, source)
    if folder is None:
        raise FileNotFoundError("no images to check for %s" % Path(pipeline_dir).name)
    channels = metadata.get("channels", {})
    background = channels.get("background_channel")
    channel = background if background is not None else channels.get("signal_channel", 0)
    voxel = metadata.get("voxel_size_um", {})
    vz, vxy = float(voxel.get("z", 4.0)), float(voxel.get("x", 4.0))
    return {"source": source_name, "stack": Path(folder) / ("ch%s" % channel),
            "channel": channel, "voxel_um": (vz, vxy, vxy),
            "orientation": metadata.get("orientation", "iar")}


def register_check(brain: str, summary: Dict, stack_folder) -> None:
    """Enter the picture in MouseBrain's analysis registry with what made it
    (the registry rule: every figure this tool produces is registered, with its
    method and source). Never raises: a registry problem must not stop a
    registration run; it is printed."""
    try:
        from mousebrain.analysis_registry import AnalysisRegistry
        AnalysisRegistry(analysis_name=REGISTRY_ANALYSIS).register_output(
            sample=brain, category="qc",
            files={"qc_figure": summary["png"]},
            results={k: summary[k] for k in ("declared_orientation", "atlas", "atlas_orientation",
                                             "brain_extent_mm", "atlas_extent_mm")},
            method_params={"check": "brainreg check_orientation as mean projections",
                           "orientation": summary["declared_orientation"],
                           "atlas": summary["atlas"],
                           "sampled_resolution_um": summary["sampled_resolution_um"],
                           "version": CHECK_VERSION},
            source_files={"stack": str(stack_folder)},
        )
    except Exception as e:
        print("  [!] orientation check not entered in the registry: %s" % e)


def check_brain(pipeline_dir, orientation: Optional[str] = None, atlas: str = DEFAULT_ATLAS,
                out_png=None, source: str = "auto", register: bool = True) -> Dict:
    """Draw the orientation check for one brain's pipeline folder.

    Uses the stack, channel, voxel size and orientation registration uses (or
    will use). ``orientation`` tries another one instead; its picture is then
    named QC_orientation_check_<orientation>.png, so the declared one is never
    overwritten by a trial. Written into the brain's 3_Registered_Atlas folder
    unless ``out_png`` is given, and entered in the registry."""
    pipeline_dir = Path(pipeline_dir)
    inputs = brain_inputs(pipeline_dir, source)
    declared = inputs["orientation"]
    use = orientation or declared
    if out_png is None:
        name = CHECK_FILENAME if use == declared else "QC_orientation_check_%s.png" % use
        out_png = pipeline_dir / "3_Registered_Atlas" / name
    title = "%s -- %s images, registration channel ch%s%s" % (
        pipeline_dir.name, inputs["source"], inputs["channel"],
        "" if use == declared else " (TRIAL: declared orientation is '%s')" % declared)
    summary = orientation_check(inputs["stack"], use, inputs["voxel_um"], atlas, out_png,
                                title=title)
    if register:
        register_check(pipeline_dir.name, summary, inputs["stack"])
    return summary


def find_brain(name: str) -> Optional[Path]:
    """A brain's pipeline folder from its name (an exact folder name first,
    then a unique partial match), or the path itself when one is given."""
    p = Path(name)
    if p.is_dir():
        return p
    from mousebrain.config import BRAINS_ROOT
    root = Path(BRAINS_ROOT)
    folders = [f for m in root.iterdir() if m.is_dir() for f in m.iterdir() if f.is_dir()]
    exact = [f for f in folders if f.name == name]
    if exact:
        return exact[0]
    partial = [f for f in folders if name in f.name]
    return partial[0] if len(partial) == 1 else None


def main(argv=None) -> int:
    """mousebrain-orientation-check: draw BrainGlobe's orientation check for a brain."""
    import argparse
    ap = argparse.ArgumentParser(
        prog="mousebrain-orientation-check",
        description="Draw BrainGlobe's orientation check for a brain as a picture: the "
                    "stack turned into the atlas's orientation as declared, averaged along "
                    "each axis, under the same views of the atlas. Written to the brain's "
                    "3_Registered_Atlas folder as QC_orientation_check.png (a trial "
                    "orientation as QC_orientation_check_<orientation>.png) and entered in "
                    "MouseBrain's analysis registry. Registration runs it by itself.")
    ap.add_argument("brain", help="brain pipeline folder name (e.g. 349_CNT_01_02_1p625x_z4) or path")
    ap.add_argument("--orientation", help="try this orientation instead of the declared one "
                                          "(three letters, e.g. sar)")
    ap.add_argument("--atlas", default=DEFAULT_ATLAS, help="registration atlas (default %(default)s)")
    ap.add_argument("--source", default="auto", help="which images: auto, manual, cropped or full")
    ap.add_argument("--out", help="write the picture here instead")
    ap.add_argument("--no-register", action="store_true", help="do not enter it in the registry")
    args = ap.parse_args(argv)
    folder = find_brain(args.brain)
    if folder is None:
        print("[FAIL] no single brain folder matches %r" % args.brain)
        return 1
    try:
        s = check_brain(folder, args.orientation, args.atlas, args.out, args.source,
                        register=not args.no_register)
    except Exception as e:
        print("[FAIL] %s" % e)
        return 1
    print("[OK] %s" % s["png"])
    print("     declared orientation '%s'; brain extent %s mm vs atlas %s mm (front-back, "
          "top-bottom, left-right)" % (s["declared_orientation"], s["brain_extent_mm"],
                                       s["atlas_extent_mm"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
