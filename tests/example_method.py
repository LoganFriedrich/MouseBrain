"""An invented approved method for tests.

The tool has no built-in approved method (analysis_registry: nothing is
approved until a lab writes its approved-method file), so a test that needs
one -- to register an entry, write an approved_method.json, or compare
hashes -- takes this one. The values are test data, not anyone's choice.
"""

EXAMPLE_METHOD = {
    "detection": "threshold+log",
    "threshold_fraction": 0.20,
    "log_threshold": 0.005,
    "log_decision_tree": True,
    "size_filter_um": [8, 25],
    "colocalization": "background_mean",
    "sigma_threshold": 0,
    "soma_dilation": 6,
    "background": "gmm",
    "background_percentile": 10,
    "bg_exclusion_dilation": 50,
}


def write_example_approved_method(folder, method=None):
    """Write *method* (EXAMPLE_METHOD by default) as approved_method.json in
    *folder* -- the registry root a test points MOUSEBRAIN_REGISTRY_ROOT at --
    and return the file's path. For tests about something else that need the
    batch to have an approved method to run with."""
    import json
    from pathlib import Path
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "approved_method.json"
    path.write_text(json.dumps(dict(method or EXAMPLE_METHOD)), encoding="utf-8")
    return path


# Every value process_single() needs, for tests that call it directly (it has
# no built-in values). Test data only.
PROCESS_SINGLE_VALUES = dict(
    threshold=0, sigma_threshold=0, dilation=50, bg_percentile=10,
    soma_dilation=6, threshold_fraction=0.20, log_threshold=0.005,
    min_diameter_um=8.0, max_diameter_um=25.0,
)
