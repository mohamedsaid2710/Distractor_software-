"""PCIbex item lines must be valid JS string literals.

The old formatter escaped double quotes but not backslashes, so a backslash in
a stimulus closed the literal early and PCIbex parsed the rest of the line as
code -- corrupting the entire item list, not one item.  Embedded newlines from
quoted CSV fields did the same.
"""
import json
import re

import pytest

from output import ibex_line

# Extracts the s:"..." and a:"..." payloads, honouring backslash escapes.
FIELD_RE = re.compile(r'(?<![\w])([sa]):("(?:[^"\\]|\\.)*")')


def _fields(line):
    return {k: json.loads(v) for k, v in FIELD_RE.findall(line)}


def test_ordinary_stimulus_round_trips():
    line = ibex_line("filler", "1", "The dog barked.", "x-x-x cat ran.")
    assert _fields(line) == {"s": "The dog barked.", "a": "x-x-x cat ran."}


def test_output_is_unchanged_for_clean_text():
    """No gratuitous diff against previously generated files."""
    line = ibex_line("filler", "1", "The dog barked.", "x-x-x cat ran.")
    assert line == '[["filler", \'1\'], "Maze", {s:"The dog barked.", a:"x-x-x cat ran."}], \n'


@pytest.mark.parametrize("text", [
    'a\\"b',                 # the reported corruption case
    "back\\slash",
    'quote"inside',
    "line1\nline2",
    "tab\there",
    'both\\and"',
])
def test_hostile_text_survives_a_round_trip(text):
    line = ibex_line("t", "1", text, "x-x-x y")
    assert _fields(line)["s"] == text


def test_line_has_no_raw_newline_inside_the_literal():
    """A raw newline would terminate the JS statement mid-string."""
    line = ibex_line("t", "1", "line1\nline2", "x-x-x y")
    assert line.count("\n") == 1
    assert line.endswith("\n")


def test_tag_is_escaped_too():
    line = ibex_line('od"d', "1", "a b", "x-x-x y")
    assert line.startswith('[["od\\"d", ')


def test_ibexify_and_generator_agree(tmp_path):
    """One formatter, so the two paths cannot drift apart."""
    from ibexify import ibexify
    src = tmp_path / "in.txt"
    src.write_text('rel;1;He said "hi".;x-x-x qq ww\n', encoding="utf-8")
    dst = tmp_path / "out.txt"
    ibexify(str(src), str(dst))
    assert dst.read_text(encoding="utf-8") == ibex_line(
        "rel", "1", 'He said "hi".', "x-x-x qq ww")
