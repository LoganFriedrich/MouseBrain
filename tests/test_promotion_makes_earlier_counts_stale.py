"""Counts made before a promotion are shown stale after it.

Promotion copied the run's files with their own (older) modification time, so
counts made after that run but before the promotion looked newer than the
promoted cells.xml and were never shown stale. Identifiers are invented.
"""
import os
import time
from pathlib import Path

from mousebrain.classification_runs import promote_to_production
from mousebrain.pipeline_status import stale_outputs


def test_counts_made_before_the_promotion_are_stale(tmp_path):
    brain = tmp_path / "831_PROJ_08_31_2p5x_z5"
    run = tmp_path / "optimization_runs" / brain.name / "class_20000101_000000"
    run.mkdir(parents=True)
    (run / "cells.xml").write_text("<Marker></Marker>")
    long_ago = time.time() - 3600
    os.utime(run / "cells.xml", (long_ago, long_ago))  # the run was made an hour ago

    counts = brain / "6_Region_Analysis" / "cell_counts_by_region.csv"
    counts.parent.mkdir(parents=True)
    counts.write_text("region_acronym,cell_count\nAAA,1\n")
    half_hour_ago = time.time() - 1800
    os.utime(counts, (half_hour_ago, half_hour_ago))  # counted after the run, before promotion

    promote_to_production(run, brain, exp_id="cla_9")

    cells = brain / "5_Classified_Cells" / "cells.xml"
    assert cells.stat().st_mtime > half_hour_ago
    assert "count" in stale_outputs(brain)
