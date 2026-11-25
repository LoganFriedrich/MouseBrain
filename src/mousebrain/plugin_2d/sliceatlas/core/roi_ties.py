"""Keep a sample's ROI counts tied to the detection they were counted against.

The ROI counter (cli/run_roi.py) registers each sample's counts in
``<project>_ROI_Analysis`` with the measurements file it counted, that
file's SHA-256 (``source_files.measurements_sha256``) and the detection
entry's method. Checking that tie only at counting time would leave ROI
counts current in the region summary after the detection batch registers
NEW measurements for the sample (a re-run with the same settings, a
corrected re-export of the image), finds no nuclei in it, or retires its
detection entry after a failure: they would be counted against measurements
that are no longer the current ones, until someone happened to count again.

retire_roi_counts_of() is called by the detection batch at each of those
moments. It retires the sample's ROI entries (AnalysisRegistry.invalidate:
files to _archived/<time>/, the row taken out of roi_summary_<region>.csv,
an invalidate_entry log line holding the counts) unless the new detection
registered measurements with exactly the bytes the counts were made from.
Entries that record no fingerprint (registered by an ROI producer that did
not record one) are retired only when the detection is known to
have changed -- its newly registered measurements differ from the ones it
registered before -- or no longer has measurements: what they were counted
against is unknown, and an unchanged re-run must not retire them.

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


def registered_measurements_sha(detection_registry, sample: str) -> Optional[str]:
    """SHA-256 of the measurements file *sample*'s CURRENT detection entry
    registers, or None when there is none or it cannot be read (never
    raises: it only decides about counts that record no fingerprint, and
    None keeps those as they are)."""
    try:
        entry = detection_registry.get_entry(sample)
        if not entry or not entry.get("is_current", True):
            return None
        rel = (entry.get("outputs") or {}).get("measurements")
        if not rel:
            return None
        path = Path(rel)
        if not path.is_absolute():
            path = Path(detection_registry.registry_root) / path
        return file_sha256(path)
    except Exception:
        return None


def retire_roi_counts_of(registry_root, project: str, sample: str,
                         new_measurements_sha: Optional[str] = None,
                         previous_measurements_sha: Optional[str] = None,
                         ) -> Tuple[Optional[bool], str]:
    """Retire *sample*'s current ROI counts when the detection they rest on
    is no longer the current one.

    *new_measurements_sha* is the SHA-256 of the measurements the detection
    batch has just registered for the sample; None means the sample now has
    no current measurements (no nuclei, or its detection entry was retired),
    and the counts are retired whatever they were counted against.
    *previous_measurements_sha* is the SHA-256 of the measurements the
    sample's detection entry registered before this run (None when unknown);
    it decides only for counts that record no fingerprint of their own.

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
    # Every current entry of the sample: the counts (key <sample>__roi_counts)
    # and any entry an earlier ROI producer registered under the sample's
    # own name.
    current = [e for e in roi.get_all_entries().values()
               if e.get("sample") == sample and e.get("is_current", True)]

    def still_rests_on_the_detection(e):
        if new_measurements_sha is None:
            return False
        counted = (e.get("source_files") or {}).get("measurements_sha256")
        if counted:
            return counted == new_measurements_sha
        # The entry records no fingerprint, so what it was counted against
        # is unknown. It is retired only when
        # the detection itself is known to have changed (the measurements
        # registered before this run differ from the new ones); an unchanged
        # re-run, or a first registration, leaves it as it was.
        return (previous_measurements_sha is None
                or previous_measurements_sha == new_measurements_sha)

    if all(still_rests_on_the_detection(e) for e in current):
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
