"""regenerate_derived_counts: eLife and laterality tables derived from region_counts.csv."""
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pipeline" / "regenerate_derived_counts.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("regenerate_derived_counts", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fake_aggregate(counts):
    """Two groups: A takes acronyms starting with 'a', B the rest; unmapped for 'z*'."""
    out = {}
    for acr, n in counts.items():
        g = '[Unmapped]' if acr.startswith('z') else ('Group A' if acr.startswith('a') else 'Group B/C')
        out.setdefault(g, {'count': 0})
        out[g]['count'] += n
    return out


GROUPS = ['Group A', 'Group B/C']


def _write_region_counts(path):
    rows = [
        {'brain': '349_CNT_01_02_1p625x_z4', 'run_date': '2026-01-01', 'total_cells': '30', 'total_left': '10',
         'total_right': '20', 'region_ab': '12', 'region_bc': '15', 'region_zz': '3',
         'region_left_ab': '4', 'region_right_ab': '8', 'region_left_bc': '6', 'region_right_bc': '9'},
        {'brain': '350_CNT_01_03_1p625x_z4', 'run_date': '2026-01-02', 'total_cells': '5', 'total_left': '',
         'total_right': '', 'region_ab': '5', 'region_bc': '', 'region_zz': ''},
    ]
    cols = sorted({k for r in rows for k in r})
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})


def test_elife_rows_and_columns(mod):
    rows = [{'brain': '349_CNT_01_02_1p625x_z4', 'total_cells': '30', 'region_ab': '12', 'region_bc': '15',
             'region_zz': '3', 'region_left_ab': '4', 'region_right_ab': '8'}]
    out, cols = mod.build_elife_rows(rows, _fake_aggregate, GROUPS)
    r = out[0]
    assert r['brain_id'] == '349'
    assert r['group_Group_A'] == 12 and r['group_Group_B_C'] == 15 and r['group_Unmapped'] == 3
    assert r['group_left_Group_A'] == 4 and r['group_right_Group_A'] == 8
    assert cols[:len(mod.FIXED_COLUMNS)] == mod.FIXED_COLUMNS
    assert cols.index('group_Group_A') < cols.index('group_left_Group_A') < cols.index('group_right_Group_A')


def test_laterality_rows_skip_brains_without_hemispheres(mod):
    rows = [{'brain': '349_CNT_01_02_1p625x_z4', 'total_left': '10', 'total_right': '20',
             'region_left_ab': '4', 'region_right_ab': '8', 'region_left_bc': '6', 'region_right_bc': '9'},
            {'brain': '350_CNT_01_03_1p625x_z4', 'total_left': '', 'total_right': ''}]
    lat = mod.build_laterality_rows(rows, _fake_aggregate, GROUPS)
    assert {r['brain_id'] for r in lat} == {'349'}
    a = next(r for r in lat if r['elife_group'] == 'Group A')
    assert (a['left'], a['right'], a['dominant']) == (4, 8, 'R')
    assert a['laterality_index'] == round((4 - 8) / 12, 3)


def test_regenerate_writes_and_archives(mod, tmp_path, monkeypatch):
    import types
    fake = types.ModuleType("mousebrain.region_mapping")
    fake.aggregate_to_elife = _fake_aggregate
    fake.ELIFE_GROUPS = GROUPS
    monkeypatch.setitem(sys.modules, "mousebrain.region_mapping", fake)
    _write_region_counts(tmp_path / 'region_counts.csv')
    (tmp_path / 'elife_counts.csv').write_text("brain,group_Group_A\nold_brain,1\n", encoding='utf-8')

    dry = mod.regenerate(tmp_path, dry_run=True, log=lambda *a: None)
    assert dry['written'] is False and not (tmp_path / 'elife_counts_archive.csv').exists()

    res = mod.regenerate(tmp_path, log=lambda *a: None)
    assert res['written'] and res['elife_rows'] == 2 and res['archived'] == 1
    archived = list(csv.DictReader(open(tmp_path / 'elife_counts_archive.csv', encoding='utf-8')))
    assert archived[0]['brain'] == 'old_brain' and archived[0]['archived_at']
    new = list(csv.DictReader(open(tmp_path / 'elife_counts.csv', encoding='utf-8')))
    assert [r['brain_id'] for r in new] == ['349', '350']
    lat = list(csv.DictReader(open(tmp_path / 'hemisphere_laterality_analysis.csv', encoding='utf-8')))
    assert len(lat) == len(GROUPS)  # only the brain with hemisphere data
