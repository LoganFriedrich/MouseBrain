"""An unconfigured install must not create files where it happens to be standing.

The root used to be a RELATIVE stand-in, `Path("CONNECTOME_ROOT_NOT_CONFIGURED")`. That
is safe for code which checks `exists()` before reading -- and ten places in this
package instead call `mkdir(parents=True, exist_ok=True)`, which checks nothing. So an
unconfigured run created
`<cwd>/CONNECTOME_ROOT_NOT_CONFIGURED/3_Nuclei_Detection/2_Data_Summary/` containing a
brand-new empty calibration_runs.csv. On 2026-10-07 that landed at the root of the lab's
share.

The stray folder was never the real danger. A SECOND empty tracker is: it looks like a
working one, later runs append to it, and the actual record of every calibration run is
orphaned while nothing appears to be wrong.

These run in-process and force the unconfigured state, because auto-detection succeeds
whenever the package itself sits inside a pipeline layout -- so simply unsetting the
environment variable does not reproduce it.
"""
import importlib
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def unconfigured(monkeypatch):
    """config as it behaves on a machine with nothing set and nothing detectable."""
    from mousebrain import config as cfg
    monkeypatch.delenv("CONNECTOME_ROOT", raising=False)
    monkeypatch.delenv("SCI_CONNECTOME_ROOT", raising=False)
    monkeypatch.setattr(cfg, "_find_repo_root", lambda: None)
    with pytest.warns(UserWarning):
        root = cfg._get_root_path()
    monkeypatch.setattr(cfg, "ROOT_PATH", root)
    return cfg, root


def test_the_stand_in_root_is_absolute(unconfigured):
    cfg, root = unconfigured
    assert root.is_absolute(), (
        "a relative stand-in resolves against the working directory, which is how "
        "this wrote into the root of the share")


def test_the_stand_in_root_is_somewhere_harmless(unconfigured):
    cfg, root = unconfigured
    assert Path(tempfile.gettempdir()) in root.parents or root.parent == Path(tempfile.gettempdir()), (
        "if something writes anyway it must write somewhere self-cleaning, not into a "
        "share, a repository, or the user's current folder")
    assert not root.exists(), "the stand-in must not be created merely by asking for it"


def test_is_configured_is_false_when_nothing_resolves(unconfigured):
    cfg, _ = unconfigured
    assert cfg.is_configured() is False


def test_is_configured_is_true_for_a_real_root(monkeypatch, tmp_path):
    from mousebrain import config as cfg
    monkeypatch.setattr(cfg, "ROOT_PATH", tmp_path)
    assert cfg.is_configured() is True


def test_the_tracker_refuses_to_fork_itself_when_unconfigured(unconfigured, tmp_path, monkeypatch):
    """The specific harm: a second empty tracker that later runs append to."""
    cfg, root = unconfigured
    from mousebrain import tracker as tk
    bogus = root / "3_Nuclei_Detection" / "2_Data_Summary" / "calibration_runs.csv"
    monkeypatch.setattr(tk, "DEFAULT_TRACKER_PATH", bogus)
    with pytest.raises(RuntimeError) as e:
        tk.ExperimentTracker()
    assert "CONNECTOME_ROOT" in str(e.value), "the refusal must say how to fix it"
    assert not bogus.exists() and not bogus.parent.exists(), (
        "and it must not have created anything on the way out")


def test_an_explicit_path_still_works_when_unconfigured(unconfigured, tmp_path):
    """Naming a path on purpose is not a guess, so it stays allowed -- tests and
    one-off analyses rely on it."""
    cfg, _ = unconfigured
    from mousebrain.tracker import ExperimentTracker
    target = tmp_path / "somewhere" / "runs.csv"
    t = ExperimentTracker(csv_path=target)
    assert t.csv_path.exists()


def test_a_configured_tracker_is_untouched(monkeypatch, tmp_path):
    """The guard must only bite the unconfigured case."""
    from mousebrain import config as cfg, tracker as tk
    monkeypatch.setattr(cfg, "ROOT_PATH", tmp_path)
    target = tmp_path / "2_Data_Summary" / "calibration_runs.csv"
    monkeypatch.setattr(tk, "DEFAULT_TRACKER_PATH", target)
    tk.ExperimentTracker()
    assert target.exists()
