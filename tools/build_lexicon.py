#!/usr/bin/env python3

"""Build the read-only POS lexicon artifact for one language.

Why this exists
---------------
POS tags used to live in a cache that generation itself wrote to, at the end of
every run and again mid-run.  The write was append-only -- "if a word is
already in the cache on disk, do NOT overwrite it" -- which made a wrong tag
permanent: no amount of re-running could correct it, and each run's tagging fed
the next run's candidate pools, so the same input with the same seed could
produce different output.

Measured on the shipped German cache, 14 of 17 proper nouns that leaked into a
sample run were recorded there as ADV/VERB/NUM/ADJ while the current tagger
calls every one of them PROPN.  Those entries could not be fixed in place.

So the lexicon is now an artifact: generation only ever reads it, and it is
rebuilt deliberately, here.  Each build is stamped with its source, licence,
citation and date, so an output file can be traced to the lexicon that
produced it.

Source
------
Each word's DOMINANT part of speech, and the share of its uses that reading
accounts for, come from published lexical norms; see src/maze_automate/lexicons/norms.py for why
and for the citations.  Tagging bare words with a tagger (the old method) was
removed: taggers need context, and on a bare word they guess.

Getting the norms (into data/norms/, which is gitignored)
---------------------------------------------------------
    # English: SUBTLEX-US with PoS (no stated licence; research use), ~11 MB
    mkdir -p data/norms/en
    curl -L -o data/norms/en/subtlex_us_pos.xlsx https://osf.io/download/7wx25/
    # German: dlexDB (CC BY-SA 4.0), ~2.2 GB -- version 2; version 1
    # (record 15097664) has a truncated POS table
    mkdir -p data/norms/de
    curl -L -o data/norms/de/dlex_v2.zip "https://zenodo.org/records/15264903/files/dlex.zip?download=1"
    # Arabic: CAMeL Tools data (GPL v2), ~540 MB into ~/.camel_tools
    uv run camel_data -i light

Usage
-----
    uv run python tools/build_lexicon.py --lang en
    uv run python tools/build_lexicon.py --lang de
    uv run python tools/build_lexicon.py --lang ar
    uv run python tools/build_lexicon.py --lang de --dry-run     # report, write nothing

A build overwrites the lexicon file; the previous one can be restored from git.
"""

import argparse
import collections
import datetime
import json
import os
import sys

import os as _os
import sys as _sys

# Make src/maze_automate importable when run as `python tools/<script>.py`.
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))

from maze_automate.params import set_params
from maze_automate.utils import REPO_ROOT, CONFIG_DIR, lexicon_path
from maze_automate.lexicons import norms as norms_mod

LANGS = {
    "de": {
        "params": os.path.join(CONFIG_DIR, "params_de.txt"),
        "dict_class": "wordfreq_German_zipf_dict",
        "out": lexicon_path("de"),
    },
    "en": {
        "params": os.path.join(CONFIG_DIR, "params_en.txt"),
        "dict_class": "wordfreq_English_zipf_dict",
        "out": lexicon_path("en"),
    },
    "ar": {
        "params": os.path.join(CONFIG_DIR, "params_ar.txt"),
        "dict_class": "wordfreq_Arabic_zipf_dict",
        "out": lexicon_path("ar"),
    },
}


def spacy_norp(words, model="en_core_web_lg"):
    """Words spaCy's NER calls NORP: nationalities, religious and political groups.

    Demonyms and geopolitical adjectives -- italian, soviet, libyan, indonesian,
    iraqi, brits, jews -- are a class no POS filter removes, because they are
    adjectives rather than proper nouns. They are poor Maze distractors for the
    same reason a proper noun is (they stand out), and some are loaded terms
    that have no place in a stimulus set.

    NORP is the label built for exactly this category. Measured on a probe of 14
    demonyms and 9 ordinary words it caught 12 of the 14 with ZERO false
    positives, so the collateral cost on the general vocabulary is negligible.
    It does not catch everything (`scots`, `isis` slipped through in a neutral
    frame); `config/exclude_en.txt` remains the mechanism for individual words.

    Returns the set of NORP words, or an empty set if spaCy is unavailable.
    """
    try:
        import spacy
    except ImportError:
        print("    [BUILD] spacy not installed; skipping the NORP check.",
              file=sys.stderr)
        return set()
    try:
        nlp = spacy.load(model)
    except Exception as e:
        print(f"    [BUILD] could not load {model} ({e}); skipping the NORP check.",
              file=sys.stderr)
        return set()

    print(f"    [BUILD] NORP check: spaCy NER on {len(words)} words...", flush=True)
    out = set()
    frames = [f"I saw {w} yesterday." for w in words]
    for w, doc in zip(words, nlp.pipe(frames, batch_size=256)):
        tok = doc[2] if len(doc) > 2 else None
        if tok is None:
            continue
        if any(e.label_ == "NORP" and e.start <= tok.i < e.end for e in doc.ents):
            out.add(w)
    return out


def _write_artifact(artifact, out_path, spec, dry_run):
    if dry_run:
        print(">>> --dry-run: nothing written")
        return 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, out_path)
    print(f">>> wrote {out_path}")
    return 0


def build_from_norms(lang, words, out_path, spec, dry_run, second_opinion):
    """Dominant POS + share per candidate word, from lexical norms."""
    from maze_automate import candidates
    src = norms_mod.SOURCES[lang]
    print(f">>> source: {src['name']} ({src['licence']})")
    spellings = {}
    norms = norms_mod.load_norms(lang, words, REPO_ROOT,
                                     stts_to_upos=candidates.STTS_TO_UPOS,
                                     spellings=spellings)

    # German: wordfreq writes every ß as ss ("heissem").  Entries are keyed by
    # the real spelling ("heißem"), and `spellings` tells the run which
    # wordfreq words to rewrite, so the word that is scored, length-matched
    # and printed is the correctly spelled one.
    tags, shares = {}, {}
    for w in words:
        if w in norms:
            key = spellings.get(w, w)
            tags[key], share = norms[w]
            if share is not None:
                shares[key] = round(share, 4)
    if spellings:
        print(f">>> {len(spellings)} words get their ß back "
              f"(e.g. {', '.join(sorted(spellings.values())[:5])})")
    coverage = len(tags) / len(words) if words else 0
    print(f">>> covered {len(tags)}/{len(words)} candidate words ({coverage:.1%}); "
          f"the rest stay untagged, so pos_match >= 1 never uses them")

    # Demonyms are adjectives in every norm; keep excluding them as before.
    if second_opinion and lang == 'en':
        norp = spacy_norp(sorted(tags))
        promoted = sum(1 for w in norp if tags.get(w) not in (None, 'PROPN'))
        for w in norp:
            if w in tags:
                tags[w] = 'PROPN'
        print(f">>> NORP check marked {promoted} demonyms/group terms as PROPN")

    clitics, misspelled = [], {}
    if lang == 'ar':
        # Candidates carrying an attached w+/f+/b+/k+/l+ are removed from the
        # Arabic vocabulary at run time; deciding it here means the run needs
        # neither CAMeL's data nor a segmenter.
        clitics = sorted(norms_mod.camel_proclitic_words(words))
        print(f">>> {len(clitics)} words carry a w+/f+/b+/k+/l+ proclitic "
              f"and will not be used as distractors")
        # Web misspellings (فتاه, هذة, بى) get no entry and are left out
        # of the vocabulary at run time.
        misspelled = norms_mod.arabic_misspellings(words)
        for w in misspelled:
            tags.pop(w, None)
            shares.pop(w, None)
        print(f">>> {len(misspelled)} misspelled words left out "
              f"(e.g. {', '.join(f'{k}->{v}' for k, v in list(misspelled.items())[:5])})")

    counts = collections.Counter(tags.values())
    print(">>> tag counts:", dict(counts.most_common()))
    for t in (0.6, 0.7, 0.8, 0.9):
        n = sum(1 for w in tags if shares.get(w, 1.0) >= t)
        print(f">>> dominant reading >= {t:.0%} of uses: {n} words")

    artifact = {
        "_meta": {
            "language": lang,
            "source_kind": "norms",
            "source": src["name"],
            "licence": src["licence"],
            "citation": src["citation"],
            "share": src["share"],
            "second_opinion": "spacy-norp" if (second_opinion and lang == 'en') else None,
            "built": datetime.date.today().isoformat(),
            "word_count": len(tags),
            "candidate_words": len(words),
            "params_file": os.path.relpath(spec["params"], REPO_ROOT),
            "note": "Read-only input to generation. Rebuild with tools/build_lexicon.py.",
        },
        "tags": tags,
        "shares": shares,
    }
    if lang == 'ar':
        artifact["clitics"] = clitics
        artifact["misspelled"] = dict(sorted(misspelled.items()))
    if spellings:
        artifact["spellings"] = dict(sorted(spellings.items()))
    return _write_artifact(artifact, out_path, spec, dry_run)


def build(lang, out_path=None, dry_run=False, second_opinion=True):
    spec = LANGS[lang]
    out_path = out_path or spec["out"]

    params = set_params(spec["params"])
    params["semantic_filter"] = False

    from maze_automate import candidates
    dict_class = getattr(candidates, spec["dict_class"])
    # German: start from wordfreq's own ss-forms, not an earlier build's
    # ß spellings, so the spelling decision is made afresh every time.
    dict_class.USE_SPELLINGS = False
    # Arabic: likewise start from the full list, not one an earlier build
    # already thinned (clitics, misspellings).
    dict_class.USE_LEXICON_EXCLUSIONS = False
    d = dict_class(params)

    words = sorted({w.text for w in d.words})
    print(f">>> {lang}: {len(words)} words in the candidate lexicon")

    return build_from_norms(lang, words, out_path, spec, dry_run, second_opinion)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lang", required=True, choices=sorted(LANGS),
                    help="language to build")
    ap.add_argument("--out", default=None, help="override the output path")
    ap.add_argument("--dry-run", action="store_true",
                    help="tag and report, but write nothing")
    ap.add_argument("--no-second-opinion", action="store_true",
                    help="skip the English demonym (spaCy NORP) check")
    args = ap.parse_args()
    sys.exit(build(args.lang, out_path=args.out, dry_run=args.dry_run,
                   second_opinion=not args.no_second_opinion))


if __name__ == "__main__":
    main()
