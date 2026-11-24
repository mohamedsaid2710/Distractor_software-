"""Files named in a params file must resolve from any working directory.

Three call sites used to resolve the same bare filename by hand and one had an
extra `dirname`, so it pointed at the repo's *parent*: the exclude list was
dropped with no warning, and the two exclusion checks in a single run could
disagree about which words were banned.
"""
import os

from maze_automate.params import set_params
from maze_automate.utils import CONFIG_DIR, REPO_ROOT, lexicon_path, resolve_repo_path


def test_anchors_point_into_the_repo():
    assert os.path.exists(os.path.join(REPO_ROOT, "run_pipeline.py"))
    assert os.path.exists(os.path.join(CONFIG_DIR, "params_de.txt"))
    for lang in ("en", "de", "ar"):
        assert os.path.exists(lexicon_path(lang)), lang


def test_bare_exclude_filename_resolves_from_another_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("exclude_en.txt", "exclude_de.txt", "exclude_ar.txt"):
        resolved = resolve_repo_path(name)
        assert os.path.isabs(resolved), name
        assert os.path.exists(resolved), name


def test_params_file_finds_its_exclude_list_from_another_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params = set_params(os.path.join(CONFIG_DIR, "params_de.txt"))
    assert params["exclude_words"] == os.path.join(CONFIG_DIR, "exclude_de.txt")


def test_list_next_to_a_copied_params_file_wins(tmp_path):
    src = open(os.path.join(CONFIG_DIR, "params_en.txt"), encoding="utf-8").read()
    (tmp_path / "my_params.txt").write_text(src, encoding="utf-8")
    (tmp_path / "exclude_en.txt").write_text("mine\n", encoding="utf-8")
    params = set_params(str(tmp_path / "my_params.txt"))
    assert params["exclude_words"] == str(tmp_path / "exclude_en.txt")


def test_absolute_path_is_returned_unchanged():
    p = "/definitely/not/here.txt"
    assert resolve_repo_path(p) == p


def test_unresolvable_name_is_returned_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_repo_path("no_such_list.txt") == "no_such_list.txt"


def test_none_passes_through():
    assert resolve_repo_path(None) is None


def test_existing_relative_path_wins(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "exclude_en.txt").write_text("local\n", encoding="utf-8")
    assert resolve_repo_path("exclude_en.txt") == "exclude_en.txt"
