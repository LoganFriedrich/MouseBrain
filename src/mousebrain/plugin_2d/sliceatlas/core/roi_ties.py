"""Keep a sample's ROI counts tied to the detection they were counted against.

The ROI counter (cli/run_roi.py) registers each sample's counts in
``<project>_ROI_Analysis`` with the measurements file it counted, that
file's SHA-256 (``source_files.measurements_sha256``) and the detection
entry's method. That tie used to be checked only at counting time. So when
the detection batch registered NEW measurements for the sample (a re-run
with the same settings, a corrected re-export of the image), found no
nuclei in it, or retired its detection entry after a failure, the ROI
counts -- counted against measurements that are no longer the current ones
-- stayed current, with their row in the region summary, until someone
happened to count again.

retire_roi_counts_of() is called by the detection batch at each of those
moments. It retires the sample's ROI entries (AnalysisRegistry.invalidate:
files to _archived/<time>/, the row taken out of roi_summary_<region>.csv,
an invalidate_entry log line holding the counts) unless the new detection
registered measurements with exactly the bytes the counts were made from.

It never creates an ROI analysis: when ``<project>_ROI_Analysis`` has no
manifest, nothing has been counted and there is nothing to retire (opening
an AnalysisRegistry creates its folders, and an empty analysis folder looks
to whoever pulls the registry like an analysis with no results). It opens
the ROI analysis without declaring whether it follows the approved method,
so the declaration its producer wrote stays as it is.
"""
import hashlib
from pathlib import Path
from typing import Optional, Tuple

from .config import roi_analysis_name


def file_sha256(path) -> Optional[str]:
    """SHA-256 of a file's bytes, or None when there is no such file."""
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def retire_roi_counts_of(registry_root, project: str, sample: str,
                         new_measurements_sha: Optional[str] = None,
                         ) -> Tuple[Optional[bool], str]:
    """Retire *sample*'s current ROI counts when the detection they rest on
    is no longer the current one.

    *new_measurements_sha* is the SHA-256 of the measurements the detection
    batch has just registered for the sample; None means the sample now has
    no current measurements (no nuclei, or its detection entry was retired),
    and the counts are retired whatever they were counted against.

    Returns (outcome, text):
      (None, "")          nothing to do: no ROI analysis, no current ROI
                          counts for the sample, or they were counted
                          against exactly the new measurements;
      (True, text)        retired; text says so;
      (False, text)       they should be retired but could not be (a file
                          would not move): they stay current; text says why.
    Raises when the ROI registry cannot be read (the caller reports it).
    """
    from mousebrain.analysis_registry import AnalysisRegistry

    name = roi_analysis_name(project)
    root = Path(registry_root)
    if not (root / "exports" / name / "registry.json").is_file():
        return None, ""
    roi = AnalysisRegistry(analysis_name=name, registry_root=root)
    entry = roi.get_entry(f"{sample}__roi_counts")
    if not entry or not entry.get("is_current", True):
        return None, ""
    counted = (entry.get("source_files") or {}).get("measurements_sha256")
    if new_measurements_sha is not None and counted == new_measurements_sha:
        return None, ""
    why = ("the detection now has different measurements"
           if new_measurements_sha is not None
           else "the detection no longer has current measurements")
    if roi.invalidate(sample=sample):
        return True, (f"ROI counts in {name} retired ({why}); count the "
                      f"sample again with brainslice-roi")
    return False, (f"ROI counts in {name} should be retired ({why}) but a file "
                   f"could not be moved; they are STILL CURRENT. Close whatever "
                   f"has them open and run the batch again")
