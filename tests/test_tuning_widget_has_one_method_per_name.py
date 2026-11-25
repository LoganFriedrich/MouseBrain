"""Setup & Tuning defines each of its methods once.

Python keeps only the LAST definition of a name in a class body, silently.
With two methods of one name (for example two ``_load_detection_results``),
the first could never run, and anyone reading or fixing it would be changing
dead code while the second one did the work. So every name is checked.
"""
import ast
import collections
from pathlib import Path

import mousebrain

SOURCE = Path(mousebrain.__file__).parent / "plugin_3d" / "tuning_widget.py"


def test_every_method_of_the_tuning_widget_is_defined_once():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    classes = [c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "TuningWidget"]
    assert classes, "TuningWidget not found"
    names = collections.Counter(
        f.name for f in classes[0].body
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)))
    assert {n: k for n, k in names.items() if k > 1} == {}
