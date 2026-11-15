"""The slice batch tool records the approved method its analysis is judged by.

WHY: approved_method.json may give one analysis its own approved method. The
manifest header then holds that method, so a producer that kept recording the
SHARED method as its entries' parameters would leave every entry "stale vs
approved" forever, however often the batch was re-run.
"""
import json
import sys
from pathlib import Path

from mousebrain.analysis_registry import DEFAULT_METHOD, AnalysisRegistry
from mousebrain.plugin_2d.sliceatlas.batch import batch_encr as be


def test_batch_detection_records_its_own_approved_method(tmp_path, monkeypatch, capsys):
    import mousebrain.config as cfg
    monkeypatch.setattr(cfg, "PIPELINE_ROOT", None)
    monkeypatch.delenv("MOUSEBRAIN_APPROVED_METHOD", raising=False)
    root = tmp_path / "Registry"
    root.mkdir()
    monkeypatch.setenv("MOUSEBRAIN_REGISTRY_ROOT", str(root))
    own = dict(DEFAULT_METHOD, threshold_fraction=0.3)
    (root / "approved_method.json").write_text(
        json.dumps({"analyses": {"ENCR_Detection": own}}), encoding="utf-8")

    raw = tmp_path / "raw" / "PRJ_01_01_HD_Regions"
    raw.mkdir(parents=True)
    (raw / "P01_01_S01_DCN.nd2").write_bytes(b"")
    out = tmp_path / "out"

    def fake_process_single(nd2_path, output_dir, **kw):
        d = Path(output_dir) / nd2_path.stem
        d.mkdir(parents=True, exist_ok=True)
        (d / "measurements.csv").write_text("a\n1\n", encoding="utf-8")
        return {"status": "completed", "n_nuclei": 3, "n_positive": 1,
                "positive_fraction": 0.33, "background": 1.0}

    monkeypatch.setattr(be, "process_single", fake_process_single)
    monkeypatch.setattr(sys, "argv", ["batch_encr", "--project", "PRJ", "--input",
                                      str(tmp_path / "raw"), "--ignore-tracker",
                                      "--output", str(out)])
    be.main()

    manifest = json.loads((root / "exports" / "ENCR_Detection" / "registry.json")
                          .read_text(encoding="utf-8"))
    entry = manifest["entries"]["P01_01_S01_DCN"]
    assert entry["method_params"] == own
    assert manifest["approved_method_hash"] == AnalysisRegistry.get_method_hash(own)
    assert entry["method_hash"] == manifest["approved_method_hash"]  # not stale
