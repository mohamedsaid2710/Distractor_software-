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
rebuilt deliberately, here.  Each build is stamped with the tagger, its
version, and the date, so an output file can be traced to the lexicon that
produced it.

Usage
-----
    python build_lexicon.py --lang de
    python build_lexicon.py --lang de --out models/german_code/german_pos_lexicon.json
    python build_lexicon.py --lang de --dry-run     # report, write nothing

The artifact is written to a NEW path and the legacy cache file is left
untouched, so a build is reversible by deleting one file.
"""

import argparse
import datetime
import importlib.metadata
import json
import os
import sys

from set_params import set_params
from utils import REPO_ROOT

LANGS = {
    "de": {
        "params": "params_de.txt",
        "dict_class": "wordfreq_German_zipf_dict",
        "out": "models/german_code/german_pos_lexicon.json",
        "legacy": "models/german_code/german_pos_cache_v2.json",
        "tagger": "HanTa",
    },
    "en": {
        "params": "params_en.txt",
        "dict_class": "wordfreq_English_zipf_dict",
        "out": "models/english_code/english_pos_lexicon.json",
        "legacy": "models/english_code/english_pos_cache.json",
        "tagger": "spaCy en_core_web_lg",
    },
    "ar": {
        "params": "params_ar.txt",
        "dict_class": "wordfreq_Arabic_zipf_dict",
        "out": "models/arabic_code/arabic_pos_lexicon.json",
        "legacy": "models/arabic_code/arabic_pos_cache.json",
        "tagger": "Farasa",
    },
}


def _tagger_version(name):
    for dist in ("HanTa", "spacy", "farasapy"):
        if dist.lower() in name.lower().replace(" ", ""):
            try:
                return f"{name} {importlib.metadata.version(dist)}"
            except Exception:
                break
    return name


def stanza_propn(lang, words, use_gpu=True):
    """Second opinion on proper-nounhood, from a different tagger.

    HanTa and Stanza are complementary here and neither is sufficient alone.
    Measured over the 34,592-word German lexicon they agree on proper-nounhood
    for 3,802 words; HanTa alone finds 535 more (alonso, daphne, kev, sobotka,
    bietigheim) and Stanza alone finds 1,514 more (itzehoe, pofalla, mclaren,
    grote, pabst, talmud).

    A proper noun that reaches the output is a giveaway -- the participant can
    pick the real word without reading it -- whereas dropping a few common
    nouns only shrinks a 15,000-word pool. So the union wins: PROPN if either
    tagger says PROPN. The cost is ~535 words HanTa mislabels (hefe, volt,
    futsal), i.e. 1.5% of the lexicon.

    Returns the set of words this tagger calls PROPN, or an empty set if
    Stanza is unavailable (the build still succeeds, with a warning).
    """
    try:
        import stanza
    except ImportError:
        print("    [BUILD] stanza not installed; skipping the second opinion. "
              "Proper-noun coverage will be lower.", file=sys.stderr)
        return set()
    try:
        nlp = stanza.Pipeline(lang, processors='tokenize,pos', use_gpu=use_gpu,
                              verbose=False, tokenize_pretokenized=True)
    except Exception as e:
        print(f"    [BUILD] could not load the Stanza {lang} pipeline ({e}); "
              f"skipping the second opinion.", file=sys.stderr)
        return set()

    print(f"    [BUILD] second opinion: Stanza {lang} on {len(words)} words...",
          flush=True)
    out = set()
    step = 5000
    for i in range(0, len(words), step):
        chunk = words[i:i + step]
        doc = nlp([[w.capitalize()] for w in chunk])
        for w, sent in zip(chunk, doc.sentences):
            if sent.words and sent.words[0].upos == 'PROPN':
                out.add(w)
        print(f"        {min(i + step, len(words))}/{len(words)}", flush=True)
    return out


def build(lang, out_path=None, dry_run=False, batch=2000, second_opinion=True):
    spec = LANGS[lang]
    out_path = out_path or os.path.join(REPO_ROOT, spec["out"])

    params = set_params(spec["params"])
    params["semantic_filter"] = False

    import wordfreq_distractor
    dict_class = getattr(wordfreq_distractor, spec["dict_class"])
    d = dict_class(params)

    words = sorted({w.text for w in d.words})
    print(f">>> {lang}: {len(words)} words in the candidate lexicon")

    if not hasattr(d, "batch_tag_words"):
        print(f"ERROR: {spec['dict_class']} has no batch_tag_words()", file=sys.stderr)
        return 1

    # Tag everything from scratch: force_refresh ignores whatever the loaded
    # cache says, which is the entire point of a rebuild.
    before = dict(d.pos_cache)
    d.pos_cache = {}
    for i in range(0, len(words), batch):
        chunk = words[i:i + batch]
        d.batch_tag_words(chunk, params=params, force_refresh=True)
        print(f"    tagged {min(i + batch, len(words))}/{len(words)}", flush=True)

    tags = {w: d.pos_cache[w] for w in words if w in d.pos_cache}

    # Fail loudly rather than writing a useless artifact.  Every batch_tag_words
    # implementation returns silently when its tagger is missing -- Arabic needs
    # Farasa (a 241 MB model plus a JVM), German needs HanTa, English needs the
    # spaCy model -- so without this check an unavailable tagger produces a
    # lexicon with zero tags, which generation would then load in place of the
    # legacy cache and run with no POS information at all.
    if not tags:
        print(f"ERROR: tagged 0 of {len(words)} words. The {lang} tagger "
              f"({spec['tagger']}) is not available, so there is nothing to "
              f"write. Install it and re-run; the existing lexicon is unchanged.",
              file=sys.stderr)
        return 1
    coverage = len(tags) / len(words)
    if coverage < 0.5:
        print(f"ERROR: only {len(tags)}/{len(words)} words ({coverage:.0%}) were "
              f"tagged. That is too sparse to replace the current lexicon -- "
              f"check that {spec['tagger']} is working. Nothing written.",
              file=sys.stderr)
        return 1

    # A full artifact can still be a useless one: when a tagger loads but fails
    # on every word, batch_tag_words records 'X' rather than raising, so the
    # lexicon comes out complete and entirely unusable ('X' enters no candidate
    # pool). An Arabic dry-run without a working Farasa produced 43,545 X and 22
    # real tags, which the emptiness check above happily passed.
    x_share = sum(1 for t in tags.values() if t == 'X') / len(tags)
    if x_share > 0.2:
        print(f"ERROR: {x_share:.0%} of tags are 'X' (unknown). {spec['tagger']} "
              f"is loading but not tagging -- an 'X' entry enters no candidate "
              f"pool, so this lexicon would be worse than none. Nothing written.",
              file=sys.stderr)
        return 1

    # German only.  Arabic gets no second opinion because neither tagger can
    # identify an Arabic proper noun in isolation: Farasa's tagset has no PROPN
    # at all (محمد and مصر both come back NOUN), and Stanza returns 'X' for
    # names AND for ordinary common nouns like بيت when given a bare word with
    # no context.  Over the whole Arabic lexicon it promoted 22 words, which is
    # noise. Arabic proper-noun exclusion needs a gazetteer, not a tagger.
    if second_opinion and lang == 'de':
        propn = stanza_propn(lang, words,
                             use_gpu=str(params.get('use_gpu', True)).lower() in ('true', '1'))
        promoted = 0
        for w in propn:
            if tags.get(w) not in (None, 'PROPN'):
                tags[w] = 'PROPN'
                promoted += 1
        if propn:
            print(f">>> second opinion promoted {promoted} words to PROPN")

    changed = sum(1 for w in tags if before.get(w) != tags[w])
    print(f">>> tagged {len(tags)} words; {changed} differ from the previous lexicon")
    counts = {}
    for t in tags.values():
        counts[t] = counts.get(t, 0) + 1
    print(">>> tag counts:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))

    artifact = {
        "_meta": {
            "language": lang,
            "tagger": _tagger_version(spec["tagger"]),
            "second_opinion": ("stanza" if (second_opinion and lang == 'de')
                               else None),
            "built": datetime.date.today().isoformat(),
            "word_count": len(tags),
            "source": "wordfreq frequency dictionary, filtered by the params file",
            "params_file": spec["params"],
            "note": "Read-only input to generation. Rebuild with build_lexicon.py.",
        },
        "tags": tags,
    }

    if dry_run:
        print(">>> --dry-run: nothing written")
        return 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=1, sort_keys=True)
    os.replace(tmp, out_path)
    print(f">>> wrote {out_path}")
    print(f">>> the legacy cache at {spec['legacy']} is untouched; "
          f"delete {out_path} to fall back to it")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lang", required=True, choices=sorted(LANGS),
                    help="language to build")
    ap.add_argument("--out", default=None, help="override the output path")
    ap.add_argument("--dry-run", action="store_true",
                    help="tag and report, but write nothing")
    ap.add_argument("--no-second-opinion", action="store_true",
                    help="skip the Stanza cross-check of proper nouns (de/ar)")
    args = ap.parse_args()
    sys.exit(build(args.lang, out_path=args.out, dry_run=args.dry_run,
                   second_opinion=not args.no_second_opinion))


if __name__ == "__main__":
    main()
