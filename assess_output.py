#!/usr/bin/env python3
"""Assess distractor output quality for English, German, and Arabic pipelines.

Checks:
- First token placeholder policy (from params).
- No placeholder tokens after position 0.
- Distractor tokens are word-like.
- Optional length matching against target tokens.
- Plausibility margin: delta = surprisal(distractor) - surprisal(target).
"""

import argparse
import csv
import importlib
import os
import re
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "input.py")) else os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from input import read_input
from set_params import set_params
from utils import strip_punct


# Latin letters, German umlauts, accented chars, and Arabic characters.
WORD_RE = re.compile(r"^[A-Za-zÄÖÜäöüßÀ-ÖØ-öø-ÿ\u0600-\u06FF]+$")
X_PLACEHOLDER_RE = re.compile(r"^x(?:-x)*$", re.IGNORECASE)

# Lazy-loaded SpaCy
_spacy_nlp = None

def get_nlp(lang):
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp
    if lang != 'de':
        return None
    try:
        import spacy
        try:
            _spacy_nlp = spacy.load("de_core_news_sm")
        except Exception:
            try:
                _spacy_nlp = spacy.load("de_core_news_md")
            except Exception:
                _spacy_nlp = None
    except ImportError:
        _spacy_nlp = None
    return _spacy_nlp

def is_x_placeholder(tok):
    return bool(X_PLACEHOLDER_RE.fullmatch(strip_punct(tok or "")))


def x_placeholder_len(tok):
    return len(re.findall(r"x", strip_punct(tok or "").lower()))


def load_output_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delim = ";"
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,")
            delim = dialect.delimiter
        except Exception:
            delim = ";" if ";" in sample else ","
        return list(csv.reader(f, delimiter=delim, quotechar='"'))


def build_model(params):
    model_loc = params.get("model_loc", "models.english_code.model")
    model_class = params.get("model_class", "EnglishScorer")
    cls = getattr(importlib.import_module(model_loc), model_class)
    return cls(params)


def score_distractor(model, sentence_obj, idx, token):
    lab = sentence_obj.labels[idx]
    target_s = sentence_obj.surprisal[lab]
    if hasattr(model, "get_surprisal_from_hidden") and lab in sentence_obj.hiddens:
        hidden = sentence_obj.hiddens[lab]
        dist_s = model.get_surprisal_from_hidden(hidden, token)
    else:
        probs = sentence_obj.probs[lab]
        dist_s = model.get_surprisal(probs, token)
    return dist_s - target_s, dist_s


def load_exclusions(params):
    exclude_path = params.get("exclude_words", None)
    if not exclude_path:
        return set()
    if not os.path.exists(exclude_path):
        return set()
    exclusions = set()
    try:
        with open(exclude_path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip().lower()
                if w and not w.startswith("#"):
                    exclusions.add(w)
    except Exception:
        pass
    return exclusions


def main():
    ap = argparse.ArgumentParser(description="Assess EN/DE/AR distractor output quality")
    ap.add_argument("-i", "--input", dest="input", required=True, help="Input source file")
    ap.add_argument("-o", "--output", dest="output", required=True, help="Generated delim output file")
    ap.add_argument("-p", "--params", dest="params", default="params_en.txt", help="Params file used for generation")
    ap.add_argument(
        "--min-delta",
        type=float,
        default=None,
        help="Required minimum (distractor surprisal - target surprisal). Defaults to value in params file.",
    )
    ap.add_argument(
        "--min-abs",
        type=float,
        default=None,
        help="Required minimum absolute surprisal. Defaults to value in params file.",
    )
    ap.add_argument(
        "--max-examples",
        type=int,
        default=12,
        help="How many failing examples to print",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if any failure is found",
    )
    args = ap.parse_args()

    params = set_params(args.params)
    model = build_model(params)
    first_token_placeholder = bool(params.get("first_token_placeholder", True))
    enforce_length_match = bool(params.get("enforce_length_match", True))
    min_delta = args.min_delta if args.min_delta is not None else float(params.get("min_delta", 0.0))
    min_abs = args.min_abs if args.min_abs is not None else float(params.get("min_abs", 0.0))
    exclusions = load_exclusions(params)
    lang = params.get("language", "de")
    nlp_sp = get_nlp(lang)

    sents = read_input(args.input)
    for ss in sents.values():
        ss.do_model(model)
        ss.do_surprisals(model)

    rows = load_output_rows(args.output)

    total_positions = 0
    placeholder_errors = 0
    nonword_errors = 0
    length_errors = 0
    plausible_errors = 0
    abs_errors = 0
    banned_errors = 0
    casing_errors = 0
    examples = []

    for row in rows:
        if len(row) < 4:
            continue
        item_id, distractor_sentence = row[1], row[3]
        if item_id not in sents:
            continue
        sentence_set = sents[item_id]
        if not sentence_set.sentences:
            continue
        tag = row[0]
        sentence_obj = next((s for s in sentence_set.sentences if s.tag == tag), sentence_set.sentences[0])

        dtoks = distractor_sentence.split()
        expected_first_len = len(strip_punct(sentence_obj.words[0]))
        if first_token_placeholder:
            if not dtoks or (not is_x_placeholder(dtoks[0])):
                placeholder_errors += 1
                if len(examples) < args.max_examples:
                    examples.append((item_id, 0, "first-token-not-placeholder", dtoks[0] if dtoks else ""))
            else:
                got_len = x_placeholder_len(dtoks[0])
                if got_len != expected_first_len:
                    placeholder_errors += 1
                    if len(examples) < args.max_examples:
                        examples.append((item_id, 0, f"first-placeholder-len-mismatch:{got_len}!={expected_first_len}", dtoks[0]))
        else:
            if not dtoks:
                placeholder_errors += 1
                if len(examples) < args.max_examples:
                    examples.append((item_id, 0, "missing-first-token", ""))
            elif is_x_placeholder(dtoks[0]):
                placeholder_errors += 1
                if len(examples) < args.max_examples:
                    examples.append((item_id, 0, "first-token-is-placeholder", dtoks[0]))
            elif enforce_length_match:
                tok0 = strip_punct(dtoks[0])
                len_tolerance = int(params.get('len_tolerance', 0)) if params else 0
                if abs(len(tok0) - expected_first_len) > len_tolerance:
                    length_errors += 1
                    if len(examples) < args.max_examples:
                        examples.append((item_id, 0, f"len-mismatch:{len(tok0)}!={expected_first_len}", tok0))

        max_i = min(len(dtoks), len(sentence_obj.words))
        for i in range(1, max_i):
            total_positions += 1
            tok = strip_punct(dtoks[i])

            if is_x_placeholder(dtoks[i]):
                placeholder_errors += 1
                if len(examples) < args.max_examples:
                    examples.append((item_id, i, "placeholder-after-first", tok))
                continue
            if (not tok) or (not WORD_RE.match(tok)):
                nonword_errors += 1
                if len(examples) < args.max_examples:
                    examples.append((item_id, i, "nonword-token", tok))
                continue
            if enforce_length_match:
                tgt_tok = strip_punct(sentence_obj.words[i])
                len_tolerance = int(params.get('len_tolerance', 0)) if params else 0
                if abs(len(tok) - len(tgt_tok)) > len_tolerance:
                    length_errors += 1
                    if len(examples) < args.max_examples:
                        examples.append((item_id, i, f"len-mismatch:{len(tok)}!={len(tgt_tok)}", tok))
                    continue

            # 1. Denylist check
            if tok.lower() in exclusions:
                banned_errors += 1
                if len(examples) < args.max_examples:
                    examples.append((item_id, i, "banned-word", tok))

            # 2. Casing check (German-specific)
            if nlp_sp and lang == 'de':
                clean_w = strip_punct(dtoks[i])
                if clean_w:
                    doc = nlp_sp(clean_w)
                    pos = doc[0].pos_ if len(doc) > 0 else None
                    is_cap = dtoks[i][0].isupper()
                    if pos in ('NOUN', 'PROPN'):
                        if not is_cap:
                            casing_errors += 1
                            if len(examples) < args.max_examples:
                                examples.append((item_id, i, f"casing-NOUN-not-capped:{pos}", dtoks[i]))
                    else:
                        # Non-nouns should be lowercase unless they are specific formal pronouns
                        if is_cap and clean_w.lower() not in ('sie', 'ihr', 'ihnen', 'ihre'):
                            casing_errors += 1
                            if len(examples) < args.max_examples:
                                examples.append((item_id, i, f"casing-NonNoun-is-capped:{pos}", dtoks[i]))

            # 3. Surprisal checks
            try:
                delta, dist_s = score_distractor(model, sentence_obj, i, dtoks[i])
                if delta <= min_delta:
                    plausible_errors += 1
                    if len(examples) < args.max_examples:
                        examples.append((item_id, i, f"low-delta:{delta:.2f}", tok))
                if dist_s < min_abs:
                    abs_errors += 1
                    if len(examples) < args.max_examples:
                        examples.append((item_id, i, f"low-abs:{dist_s:.2f}<{min_abs}", tok))
            except Exception:
                plausible_errors += 1
                if len(examples) < args.max_examples:
                    examples.append((item_id, i, "scoring-error", tok))

    print(f"positions_total={total_positions}")
    print(f"placeholder_errors={placeholder_errors}")
    print(f"nonword_errors={nonword_errors}")
    print(f"length_errors={length_errors}")
    print(f"banned_word_errors={banned_errors}")
    print(f"casing_errors={casing_errors}")
    print(f"low_abs_surprisal_errors={abs_errors}")
    print(f"low_delta_surprisal_errors={plausible_errors}")

    if total_positions > 0:
        total_errs = placeholder_errors + nonword_errors + length_errors + banned_errors + casing_errors + abs_errors + plausible_errors
        rate = 100.0 * total_errs / total_positions
        print(f"total_error_rate_pct={rate:.2f}")

    if examples:
        print("\nfailure examples:")
        for ex in examples:
            print(f"  Item {ex[0]} [pos {ex[1]}]: {ex[2]} -> '{ex[3]}'")

    failed = (placeholder_errors > 0) or (nonword_errors > 0) or (length_errors > 0) or \
             (plausible_errors > 0) or (abs_errors > 0) or (banned_errors > 0) or (casing_errors > 0)
    if args.strict and failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
