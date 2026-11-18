#!/usr/bin/env python3
"""
Subprocess script for performing the actual crop operation.
Called by the napari plugin after crop position is selected.

WHY THE NEW CROP IS BUILT BESIDE THE OLD ONE AND SWAPPED IN
A crop used to be written straight into 2_Cropped_For_Registration, plane by
plane, over the crop already there, with metadata.json last. An interrupted
re-crop (the window closed, the computer restarted) then left a folder of
mixed planes -- some cut at the new line, some at the old -- under the OLD
metadata, which every reader took for a finished crop. Now the new crop is
written to 2_Cropped_For_Registration.partial; only when every plane and the
metadata are written is the previous crop moved into
2_Cropped_For_Registration/_archive/<time>/ (kept, not deleted) and the new
one moved in, metadata.json last. An interrupted crop leaves the old crop
untouched (or, if it stops during the swap itself, a crop folder without
metadata.json, which every reader shows as "not cropped").

A re-crop makes the registration fitted to the earlier crop OUT OF DATE (see
mousebrain.registration_approval): the brain is back at "Register".

EVERY tool that writes a crop uses this rule (new_partial() + swap_in()):
this napari crop, step 2's --auto-crop, util_apply_crop_from_reference.py and
the manual crop widget's own writers. Step 2 and the reference-crop script
used to write in place and leave mixed planes when stopped, which the batches
then registered unattended.

This module imports only the standard library at the top, so the pipeline
scripts can import these two functions without loading napari or Qt.
"""

import os
import shutil
import sys
import json
from datetime import datetime
from pathlib import Path

FOLDER_EXTRACTED = "1_Extracted_Full"
FOLDER_CROPPED = "2_Cropped_For_Registration"
PARTIAL_SUFFIX = ".partial"


def new_partial(output_folder: Path) -> Path:
    """An empty ``<output_folder>.partial`` folder beside ``output_folder``
    to build a new crop in. One left there by an earlier crop that was
    stopped is unfinished work, made only from the extracted images: it is
    started again from nothing."""
    output_folder = Path(output_folder)
    partial = output_folder.with_name(output_folder.name + PARTIAL_SUFFIX)
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    return partial


def swap_in(output_folder: Path, partial: Path) -> Path:
    """Move the finished crop in ``partial`` into ``output_folder``.

    Everything already in ``output_folder`` (except its ``_archive``) goes
    first into ``output_folder/_archive/<time>/``. The new crop's
    metadata.json is moved LAST, so a reader never sees new metadata over
    planes that are not all there. Returns the archive folder, or None when
    there was nothing to archive.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    old = [p for p in output_folder.iterdir() if p.name != "_archive"]
    archive = None
    if old:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = output_folder / "_archive" / stamp
        n = 1
        while archive.exists():
            n += 1
            archive = output_folder / "_archive" / ("%s_%d" % (stamp, n))
        archive.mkdir(parents=True)
        # metadata.json FIRST: from this moment the folder is "not cropped"
        # to every reader, until the new metadata lands.
        old.sort(key=lambda p: p.name != "metadata.json")
        for p in old:
            shutil.move(str(p), str(archive / p.name))
    new = sorted(partial.iterdir(), key=lambda p: p.name == "metadata.json")
    for p in new:
        os.replace(str(p), str(output_folder / p.name))
    partial.rmdir()
    return archive


def apply_signal_choice(crop_meta: dict, crop_data: dict) -> None:
    """Write the crop tool's "which channel has the cells" answer into the
    crop's metadata, under the keys every later step reads.

    WHY: the crop window asks the person which channel shows the cells and
    puts the answer in .crop_position.json, but this process used to copy the
    extraction's metadata unchanged, so the answer was dropped and steps 3-5
    went on reading the extraction's ``channels.signal_channel``. The window
    starts on the extraction's own answer (manual_crop_widget), so a person who
    leaves the choice alone writes back what was there.

    What is written, and only when the answer CHANGES something:
      * the answer equals the metadata's own (``channels.signal_channel``,
        read by the one rule, brain_context.channel_roles): nothing in
        ``channels`` is touched -- a background of null, or a third channel,
        stays as extraction wrote it;
      * the answer is the other channel: ``channels.signal_channel`` becomes
        the answer and, if ``channels.background_channel`` was the answer,
        it becomes the previous signal channel (the two roles swap);
      * no answer (the window was left with nothing chosen, or an older
        window that did not ask): nothing is changed.
    The old top-level ``signal_channel`` / ``channels_swapped`` keys are
    removed from the crop's copy only when they disagree with the answer, so
    they cannot contradict it.
    """
    from mousebrain.brain_context import _channel_number

    signal = _channel_number(crop_data.get('signal_channel'))
    if signal is None or signal < 0:
        return
    channels = crop_meta.get('channels')
    if not isinstance(channels, dict):
        channels = {}
    current = _channel_number(channels.get('signal_channel'))
    if current is None:
        # No integer key yet: what the old keys said, else ch0 (the rule's default)
        old = _channel_number(crop_meta.get('signal_channel'))
        if old is None and crop_meta.get('channels_swapped') is not None:
            old = 1 if crop_meta.get('channels_swapped') else 0
        current = 0 if old is None else old
    if signal != current:
        crop_meta['channels'] = channels
        channels['signal_channel'] = signal
        if _channel_number(channels.get('background_channel')) == signal:
            channels['background_channel'] = current
        crop_meta['signal_channel_chosen_at_crop'] = signal
    top = _channel_number(crop_meta.get('signal_channel'))
    if top is not None and top != signal:
        crop_meta.pop('signal_channel', None)
    swapped = crop_meta.get('channels_swapped')
    if swapped is not None and (1 if swapped else 0) != signal:
        crop_meta.pop('channels_swapped', None)


def do_crop(pipeline_folder: Path):
    """Perform the crop operation based on saved crop position."""
    print(f"[DEBUG] Pipeline folder received: {pipeline_folder}")

    # Validate pipeline folder structure
    source_folder = pipeline_folder / FOLDER_EXTRACTED
    print(f"[DEBUG] Source folder: {source_folder}")
    print(f"[DEBUG] Source folder exists: {source_folder.exists()}")

    if not source_folder.exists():
        print(f"\nError: Source folder not found: {source_folder}")
        print(f"\nExpected structure:")
        print(f"  {pipeline_folder}/")
        print(f"    {FOLDER_EXTRACTED}/")
        print(f"      ch0/")
        print(f"      ch1/")
        print(f"\nPlease check the folder path.")
        return False

    # Read crop position
    crop_file = pipeline_folder / ".crop_position.json"
    if not crop_file.exists():
        print("Error: No crop position file found")
        return False

    with open(crop_file, 'r') as f:
        crop_data = json.load(f)

    y_crop = crop_data['y_crop']
    rows_kept = crop_data['rows_kept']
    total_rows = crop_data['total_rows']

    print(f"\n{'='*60}")
    print("Applying crop...")
    print(f"{'='*60}\n")
    print(f"Crop position: Y={y_crop}")
    print(f"Keeping: {rows_kept}/{total_rows} rows ({100*rows_kept/total_rows:.1f}%)\n")

    source_folder = pipeline_folder / FOLDER_EXTRACTED
    output_folder = pipeline_folder / FOLDER_CROPPED
    # Load metadata
    meta_path = pipeline_folder / FOLDER_EXTRACTED / "metadata.json"
    metadata = None
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

    # Perform the crop
    import tifffile
    # The new crop is built in <crop folder>.partial and swapped in only when
    # complete (see the module docstring). A .partial left by an earlier
    # interrupted crop is unfinished work: it is started again from nothing.
    partial = new_partial(output_folder)

    # Find all channels
    channels = sorted([d for d in source_folder.iterdir()
                     if d.is_dir() and d.name.startswith("ch")])

    total_files = sum(len(list(ch.glob("Z*.tif"))) for ch in channels)
    processed = 0

    for ch_folder in channels:
        ch_name = ch_folder.name
        print(f"Processing {ch_name}...")

        out_ch = partial / ch_name
        out_ch.mkdir(exist_ok=True)

        tiff_files = sorted(ch_folder.glob("Z*.tif"))

        for tiff_path in tiff_files:
            img = tifffile.imread(str(tiff_path))
            cropped = img[:y_crop, :]  # Keep everything BEFORE y_crop (top portion)

            out_path = out_ch / tiff_path.name
            tifffile.imwrite(str(out_path), cropped)

            processed += 1
            if processed % 50 == 0:
                print(f"  {processed}/{total_files} files processed ({100*processed/total_files:.1f}%)")

    print(f"  {processed}/{total_files} files processed (100.0%)")

    # Save metadata
    print("\nSaving metadata...")
    first_ch = channels[0]
    first_tif = sorted(first_ch.glob("Z*.tif"))[0]
    first_img = tifffile.imread(str(first_tif))
    orig_y, x = first_img.shape
    z = len(list(first_ch.glob("Z*.tif")))

    crop_meta = metadata.copy() if metadata else {}
    crop_meta['crop_y_end'] = y_crop
    crop_meta['crop_y_original'] = orig_y
    crop_meta['crop_y_new'] = y_crop
    crop_meta['crop_method'] = 'napari_plugin_manual'
    crop_meta['dimensions_cropped'] = {
        'z': z,
        'y': y_crop,
        'x': x,
    }
    apply_signal_choice(crop_meta, crop_data)

    with open(partial / "metadata.json", 'w') as f:
        json.dump(crop_meta, f, indent=2)

    # Every plane and the metadata are written: swap the new crop in.
    archive = swap_in(output_folder, partial)

    # Clean up temp file
    crop_file.unlink()

    print(f"\n{'='*60}")
    print("[OK] Crop complete!")
    print(f"{'='*60}")
    print(f"Output: {output_folder}")
    if archive:
        print(f"The previous crop was moved to: {archive}")
        print("A registration made on the previous crop is now OUT OF DATE:")
        print("the brain is back at 'Register' -- register it again.")
    print(f"\nYou can now run the next pipeline step (registration).")

    return True


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: crop_subprocess.py <pipeline_folder>")
        sys.exit(1)

    pipeline_folder = Path(sys.argv[1])
    success = do_crop(pipeline_folder)
    sys.exit(0 if success else 1)
