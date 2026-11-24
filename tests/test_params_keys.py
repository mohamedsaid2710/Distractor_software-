"""Every params key the code reads must be accepted by set_params.

set_params rejects unknown keys so that a typo in a params file stops the
run.  The other side of that contract is that a key the code actually reads
must be in KNOWN_KEYS, or a correct params file would be rejected.  This
test scans the source for `params.get("x")` / `params["x"]` and checks.
"""
import pathlib
import re

import pytest

from maze_automate.params import KNOWN_KEYS, RUNTIME_KEYS, ParamsError, set_params

ROOT = pathlib.Path(__file__).resolve().parent.parent
PATTERN = re.compile(r"""params(?:\.get\(|\[)\s*['"]([A-Za-z_][A-Za-z0-9_]*)['"]""")


def _keys_read_in_code():
    found = {}
    for path in ROOT.glob("**/*.py"):
        rel = path.relative_to(ROOT)
        if rel.parts[0] in ("tests", "wiki", ".venv") or rel.parts[0].startswith("."):
            continue
        for key in PATTERN.findall(path.read_text(encoding="utf-8")):
            found.setdefault(key, str(rel))
    return found


def test_every_key_read_in_code_is_known():
    missing = {k: f for k, f in _keys_read_in_code().items()
               if k not in KNOWN_KEYS and k not in RUNTIME_KEYS}
    assert not missing, f"add these to KNOWN_KEYS in params.py: {missing}"


@pytest.mark.parametrize("name", ["config/params_en.txt", "config/params_de.txt", "config/params_ar.txt"])
def test_shipped_params_files_load(name):
    set_params(str(ROOT / name))


def _write(tmp_path, extra):
    src = (ROOT / "config/params_de.txt").read_text(encoding="utf-8")
    p = tmp_path / "p.txt"
    p.write_text(src + "\n" + extra + "\n", encoding="utf-8")
    return str(p)


def test_misspelled_key_is_rejected_with_suggestion(tmp_path):
    with pytest.raises(ParamsError, match=r"unknown key 'min_pool_sise'.*min_pool_size"):
        set_params(_write(tmp_path, "min_pool_sise: 30"))


def test_runtime_key_is_rejected(tmp_path):
    with pytest.raises(ParamsError, match="target_zipf"):
        set_params(_write(tmp_path, "target_zipf: 4.0"))


def test_duplicate_key_is_rejected(tmp_path):
    with pytest.raises(ParamsError, match="already set on line"):
        set_params(_write(tmp_path, "seed: 1"))


def test_line_without_colon_is_rejected(tmp_path):
    with pytest.raises(ParamsError, match="expected 'key: value'"):
        set_params(_write(tmp_path, "min_pool_size 30"))


@pytest.mark.parametrize("line, fragment", [
    # bool() is True for any text, so these used to switch the setting ON
    ("write_report: no", "must be True or False"),
    ('write_report: "False"', "must be True or False"),
    ("semantic_filter: 1", "must be True or False"),
    # German decimal comma parses as the tuple (1, 0)
    ("min_delta: 1,0", "use a dot for decimals: 1.0"),
    ('min_delta: "8"', "remove the quotes"),
    ("len_tolerance: 1.5", "must be a whole number"),
    ("min_pool_size: null", "got null"),
])
def test_wrong_value_type_is_rejected(tmp_path, line, fragment):
    key = line.split(":")[0]
    src = "\n".join(l for l in (ROOT / "config/params_de.txt").read_text(encoding="utf-8").splitlines()
                    if not l.startswith(key + ":") and not l.startswith(key + " "))
    p = tmp_path / "p.txt"
    p.write_text(src + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ParamsError, match=re.escape(fragment)):
        set_params(str(p))


@pytest.mark.parametrize("line", ["seed: null", "freq_tolerance: null", "device: 0",
                                  "context_window: null"])
def test_nullable_and_device_values_are_accepted(tmp_path, line):
    key = line.split(":")[0]
    src = "\n".join(l for l in (ROOT / "config/params_de.txt").read_text(encoding="utf-8").splitlines()
                    if not l.startswith(key + ":") and not l.startswith(key + " "))
    p = tmp_path / "p.txt"
    p.write_text(src + "\n" + line + "\n", encoding="utf-8")
    set_params(str(p))


@pytest.mark.parametrize("line, fragment", [
    ("match_noun_pos: True", "replaced by pos_match"),
    ("match_casing_only: True", "was removed"),
    ("force_max_surprisal: True", "always chosen"),
    ("match_surprisal: False", "always chosen"),
    ("absolute_threshold_only: False", "always"),
    ("pool_size_factor: 2.0", "was removed"),
    ("pos_match: 3", "pos_match must be 0"),
])
def test_removed_keys_and_pos_match_range(tmp_path, line, fragment):
    key = line.split(":")[0]
    src = "\n".join(l for l in (ROOT / "config/params_de.txt").read_text(encoding="utf-8").splitlines()
                    if not l.startswith(key + ":"))
    p = tmp_path / "p.txt"
    p.write_text(src + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ParamsError, match=re.escape(fragment)):
        set_params(str(p))
