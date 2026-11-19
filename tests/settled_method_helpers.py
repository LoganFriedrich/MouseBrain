"""Test helpers: a settled classification method in a test tracker, and a
stand-in pre-filter.

Step 5 classifies only with the classification run marked Paradigm Best for
the brain's imaging setup, and always pre-filters first (2026-11-19). Tests
that exercise what step 5 does with its files settle a method here instead of
passing --model / --no-prefilter, which step 5 now refuses.
"""
from pathlib import Path

import mousebrain.prefilter as prefilter
from mousebrain.tracker import ExperimentTracker


def settle(csv_path, brain, model, method="full", cube=50, batch=32,
           tracing="descending"):
    """Log a completed classification run for `brain` and mark it Paradigm
    Best. Returns its exp_id."""
    t = ExperimentTracker(csv_path)
    if method == "selective":
        exp_id = t.log_selective_classification(
            brain=brain, model_path=str(model), tracing_type=tracing,
            status="completed", cube_size=cube, batch_size=batch)
    else:
        exp_id = t.log_classification(brain=brain, model_path=str(model),
                                      cube_size=cube, batch_size=batch, status="completed")
    assert t.mark_paradigm_best(exp_id)
    return exp_id


def fake_prefilter(monkeypatch, coords=((1, 2, 3), (4, 5, 6)), seen=None):
    """Replace the atlas pre-filter with one that keeps `coords`. When `seen`
    is a list, each call's keyword arguments are appended to it."""
    def run(**kw):
        if seen is not None:
            seen.append(kw)
        return {"stats": {"total": len(coords) + 1, "interior": len(coords), "suspicious": 1},
                "interior_coords": list(coords), "category_counts": {}}
    monkeypatch.setattr(prefilter, "prefilter_candidates", run)


def model_file(tmp_path, name="model.h5"):
    m = Path(tmp_path) / name
    m.write_text("m")
    return m
