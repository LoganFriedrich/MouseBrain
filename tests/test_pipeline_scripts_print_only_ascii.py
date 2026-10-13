"""Nothing the pipeline prints may contain a character the console cannot encode.

A Windows console is cp1252. One non-ASCII character in a print() raises
UnicodeEncodeError and takes the whole script down -- not at the end, but at the
moment it tries to say something, which is usually in the middle of work.

This is not hypothetical. On 2026-10-13 a tick character in a progress line
stopped the extraction of a 37 GB brain before it read a single slice, and the
two steps queued behind it then failed for a second, unrelated-looking reason.
The project has stated this rule for a long time; the scripts drifted anyway,
because a rule nobody checks is a suggestion.

Comments, docstrings, matplotlib text and Qt widget text are all exempt and are
deliberately not searched: those are rendered by something that handles Unicode,
and the rule is about the terminal.
"""
import ast
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parents[1] / "scripts" / "pipeline"
UTILS = Path(__file__).resolve().parents[1] / "scripts" / "utils"

# Calls whose arguments reach a console.
CONSOLE_CALLS = {"print", "write", "echo"}
LOGGER_METHODS = {"debug", "info", "warning", "error", "critical", "exception"}


def console_strings(path):
    """Every string literal this file hands to something that prints it.

    Walks the syntax tree rather than the text, so a tick inside a docstring or a
    matplotlib label is not reported: only strings actually passed to a printing
    call count.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            prints = (name in CONSOLE_CALLS
                      or (name in LOGGER_METHODS
                          and isinstance(node.func, ast.Attribute)
                          and "log" in ast.unparse(node.func.value).lower()))
            if prints:
                for piece in ast.walk(node):
                    if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                        found.append((piece.lineno, piece.value))
            self.generic_visit(node)

    Visitor().visit(tree)
    return found


def scripts():
    for folder in (PIPELINE, UTILS):
        if folder.is_dir():
            for p in sorted(folder.glob("*.py")):
                yield p


@pytest.mark.parametrize("script", list(scripts()), ids=lambda p: p.name)
def test_nothing_printed_contains_a_character_the_console_cannot_encode(script):
    offenders = []
    for line_no, text in console_strings(script):
        bad = sorted({c for c in text if ord(c) > 127})
        if bad:
            offenders.append("%s:%d %s" % (
                script.name, line_no,
                ", ".join("U+%04X %r" % (ord(c), c) for c in bad)))
    assert not offenders, (
        "These would raise UnicodeEncodeError on a cp1252 console. Use the ASCII "
        "replacements the project documents ([OK], [FAIL], [!], ->, um, x):\n  "
        + "\n  ".join(offenders))


def test_the_check_can_actually_see_a_tick(tmp_path):
    """A guard that cannot fail protects nothing."""
    sample = tmp_path / "sample.py"
    sample.write_text('print("done ✓")\n', encoding="utf-8")
    found = console_strings(sample)
    assert any(any(ord(c) > 127 for c in text) for _, text in found)


def test_a_docstring_tick_is_not_reported(tmp_path):
    """Docstrings are never encoded by the terminal, so they are not in scope."""
    sample = tmp_path / "sample.py"
    sample.write_text('"""A tick ✓ in prose."""\nprint("plain")\n', encoding="utf-8")
    found = console_strings(sample)
    assert not any(any(ord(c) > 127 for c in text) for _, text in found)
