"""The quality report sidecar.

The delim and ibex files carry only the chosen word, so a distractor that came
out of the desperation fallback is indistinguishable from one that met every
criterion.  Without the sidecar there is no way to state in a paper how many
items actually met the criteria.
"""
import csv

import pytest

from output import REPORT_COLUMNS, report_path, init_report, append_report, summarize_report


class FakeLabel:
    def __init__(self, distractor, report):
        self.distractor = distractor
        self.report = report


class FakeSentence:
    def __init__(self, words, labels):
        self.words = words
        self.labels = labels


class FakeSet:
    def __init__(self, id, sentences, labels):
        self.id = id
        self.sentences = sentences
        self.labels = labels


@pytest.fixture
def one_item():
    rep = {"pool_size": 200, "freq_widen_steps": 2, "target_zipf": 5.19,
           "band_min_zipf": 4.19, "band_max_zipf": 6.19, "pos_filter": "NOUN",
           "used_fallback": False, "achieved_surprisal": 30.0,
           "threshold": 25.0, "target_surprisal": 13.85,
           "is_placeholder": False}
    return FakeSet("1",
                   [FakeSentence(["this", "is", "a", "test."], [0, "1_L2", "2_L1", "3_L4"])],
                   {"1_L2": FakeLabel("ru", dict(rep)),
                    "2_L1": FakeLabel("wo", dict(rep)),
                    "3_L4": FakeLabel("spot", dict(rep))})


def test_sidecar_path_is_derived_from_the_output(tmp_path):
    assert report_path("out_en.txt").endswith("out_en.report.csv")
    assert report_path("noext").endswith("noext.report.csv")


def test_header_is_written(tmp_path):
    out = str(tmp_path / "o.txt")
    init_report(out)
    with open(report_path(out), encoding="utf-8") as f:
        assert next(csv.reader(f)) == REPORT_COLUMNS


def test_one_row_per_distractor_position(tmp_path, one_item):
    out = str(tmp_path / "o.txt")
    init_report(out)
    append_report(out, one_item, "en")
    with open(report_path(out), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3                       # position 0 is the placeholder
    assert [r["position"] for r in rows] == ["1", "2", "3"]
    assert [r["target"] for r in rows] == ["is", "a", "test"]


def test_length_and_frequency_deltas_are_computed(tmp_path, one_item):
    out = str(tmp_path / "o.txt")
    init_report(out)
    append_report(out, one_item, "en")
    with open(report_path(out), encoding="utf-8") as f:
        rows = {r["target"]: r for r in csv.DictReader(f)}
    assert rows["test"]["len_delta"] == "0"      # test / spot
    assert rows["a"]["len_delta"] == "1"         # a / wo
    assert rows["test"]["zipf_delta"] not in ("", None)


def test_threshold_compliance_is_recorded(tmp_path, one_item):
    out = str(tmp_path / "o.txt")
    init_report(out)
    append_report(out, one_item, "en")
    with open(report_path(out), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert all(r["meets_threshold"] == "True" for r in rows)   # 30.0 >= 25.0


def test_a_label_is_reported_once_even_across_condition_rows(tmp_path, one_item):
    """One distractor serves every condition row; it must not be double-counted."""
    one_item.sentences.append(
        FakeSentence(["this", "is", "a", "test."], [0, "1_L2", "2_L1", "3_L4"]))
    out = str(tmp_path / "o.txt")
    init_report(out)
    append_report(out, one_item, "en")
    with open(report_path(out), encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 3


def test_summary_counts_match_the_rows(tmp_path, one_item, capsys):
    out = str(tmp_path / "o.txt")
    init_report(out)
    append_report(out, one_item, "en")
    stats = summarize_report(out)
    assert stats["n"] == 3
    assert stats["met_threshold"] == 3
    assert stats["used_fallback"] == 0
    assert "QUALITY REPORT" in capsys.readouterr().out


def test_summary_of_a_missing_sidecar_is_empty_not_an_error(tmp_path):
    assert summarize_report(str(tmp_path / "nope.txt")) == {}
