"""The whole pipeline on deliberately awkward stimuli, in every language.

Real input parsing, real word lists and lexicons, real labels, selection,
fallback, output and report -- only the language model is a stand-in
(tests/fake_lm.py), so this runs without downloads, in CI.  It is the test
for "works on my stimuli, crashes on someone else's".
"""
import csv
import os
import re

import pytest

from maze_automate import pipeline, selection
from maze_automate.utils import CONFIG_DIR

STIMULI = {
    "en": [
        "plain;1;The lawyer who called the judge left early.",
        "plain;1;The lawyer who the judge called left early.",
        'quote;2;He said, "Stop!" and walked out of the room.',
        "short;3;Hello.",
        "short;4;Wait here.",
        "numbers;5;In 2024 she ran 5 miles before 8 am.",
        "repeat;6;The the the dog saw the cat.",
        "hyphen;7;A well-known writer didn't like the long-winded speech.",
        "acronym;8;The NASA team met the FBI agents on Monday.",
        "lengths;9;The cat slept on the old sofa.",
        "lengths;9;The elephant slept on the old sofa.",
        "long;10;" + " ".join(["The tired old farmer slowly walked home"] * 4) + ".",
    ],
    "de": [
        "plain;1;Der Anwalt, der den Richter anrief, verließ das Büro früh.",
        "plain;1;Der Anwalt, den der Richter anrief, verließ das Büro früh.",
        "quote;2;Sie sagte: „Komm sofort nach Hause!“",
        "short;3;Hallo.",
        "compound;4;Der Donaudampfschifffahrtsgesellschaftskapitän grüßte die Fahrgäste.",
        "eszett;5;Die große Straße war heiß und weiß.",
        "numbers;6;Im Jahr 2024 liefen 300 Menschen 5 Kilometer.",
        "lengths;7;Der Hund schlief.",
        "lengths;7;Der Elefant schlief.",
    ],
    "ar": [
        "plain;1;المحامي الذي اتصل بالقاضي غادر المكتب مبكرا.",
        "plain;1;المحامي الذي اتصل به القاضي غادر المكتب مبكرا.",
        "vowelled;2;ذَهَبَ الطَّالِبُ إِلَى الْمَدْرَسَةِ صَبَاحًا.",
        "punct;3;هل رأيتَ الكتاب؟ نعم، رأيته!",
        "short;4;مرحبا.",
        "tatweel;5;كـتـب الطالب درسا طويلا.",
        "numbers;6;في عام 2024 سافر 5 أشخاص.",
        "lengths;7;نام القط.",
        "lengths;7;نام الفيل.",
    ],
}


def _params_file(tmp_path, lang):
    src = open(os.path.join(CONFIG_DIR, f"params_{lang}.txt"), encoding="utf-8").read()
    src = re.sub(r"^(model_loc|model_class|semantic_filter|num_to_test|hf_model_name):.*\n",
                 "", src, flags=re.M)
    src += ('model_loc: "fake_lm"\nmodel_class: "FakeScorer"\n'
            "semantic_filter: False\nnum_to_test: 60\n")
    p = tmp_path / f"params_{lang}.txt"
    p.write_text(src, encoding="utf-8")
    return str(p)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = {}
    real_nlp = selection._get_nlp_model
    for lang, rows in STIMULI.items():
        tmp = tmp_path_factory.mktemp(lang)
        infile = tmp / "in.txt"
        infile.write_text("\n".join(rows) + "\n", encoding="utf-8")
        outfile = tmp / "out.txt"
        # English is tagged by spaCy (installed); Stanza (de/ar) would need a
        # download, so those inputs fall back to their lexicon tags.
        selection._get_nlp_model = (real_nlp if lang == "en"
                                       else (lambda *a, **k: None))
        try:
            failed = pipeline.run_stuff(str(infile), str(outfile),
                                    parameters=_params_file(tmp, lang))
        finally:
            selection._get_nlp_model = real_nlp
        with open(outfile, encoding="utf-8", newline="") as f:
            output = list(csv.reader(f, delimiter=";"))
        with open(str(outfile).rsplit(".", 1)[0] + ".report.csv", encoding="utf-8") as f:
            report = list(csv.DictReader(f))
        out[lang] = (failed, output, report)
    return out


LANGS = list(STIMULI)


@pytest.mark.parametrize("lang", LANGS)
def test_no_item_fails(runs, lang):
    assert runs[lang][0] == []


@pytest.mark.parametrize("lang", LANGS)
def test_every_input_row_has_an_output_row(runs, lang):
    assert len(runs[lang][1]) == len(STIMULI[lang])


@pytest.mark.parametrize("lang", LANGS)
def test_one_distractor_per_word(runs, lang):
    for tag, item, sentence, distractors, labels in runs[lang][1]:
        assert len(sentence.split()) == len(distractors.split()) == len(labels.split()), sentence
        assert distractors.split()[0].strip(",.;:!?\"„“«»،؛؟") == "x-x-x"


@pytest.mark.parametrize("lang", LANGS)
def test_report_covers_every_position(runs, lang):
    report = runs[lang][2]
    assert report and all(r["distractor"] for r in report)
    assert not any(r["is_placeholder"] == "True" and r["relaxation"] != "none-admissible"
                   for r in report)


@pytest.mark.parametrize("lang", LANGS)
def test_distractor_is_never_the_target(runs, lang):
    for r in runs[lang][2]:
        assert r["distractor"].lower() != r["target"].lower()
