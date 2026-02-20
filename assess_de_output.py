#!/usr/bin/env python3
"""Assess German distractor output quality against key maze constraints.

Checks:
- First distractor token is an x-placeholder matching first-word length
  (when `first_token_placeholder` is enabled).
- No later token is an x-placeholder.
- Distractors are letter-only words.
- Distractors match target token length (when `enforce_length_match` is enabled).
- Distractors are less plausible than targets in context (delta > threshold).
"""

import argparse
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from input import read_input
from set_params import set_params
from models.german_code.model import GermanScorer
from utils import strip_punct


WORD_RE = re.compile(r"^[A-Za-zÄÖÜäöüß]+$")
X_PLACEHOLDER_RE = re.compile(r"^x(?:-x)*$", re.IGNORECASE)


def is_x_placeholder(tok):
    return bool(X_PLACEHOLDER_RE.fullmatch(strip_punct(tok or "")))


def x_placeholder_len(tok):
    return len(re.findall(r"x", strip_punct(tok or "").lower()))


def load_output_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.reader(f, delimiter=";", quotechar='"'))


def main():
    ap = argparse.ArgumentParser(description="Assess German distractor output")
    ap.add_argument("--input", required=True, help="Input source file (e.g., german_sample.txt)")
    ap.add_argument("--output", required=True, help="Generated delim output file")
    ap.add_argument("--params", default="params_de.txt", help="Params file for model loading")
    ap.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Required minimum (distractor surprisal - target surprisal). Default: 0.0",
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
    model = GermanScorer(params)
    first_token_placeholder = bool(params.get("first_token_placeholder", True))
    enforce_length_match = bool(params.get("enforce_length_match", True))

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
    examples = []

    for row in rows:
        if len(row) < 4:
            continue
        tag, item_id, _sentence, distractor_sentence = row[0], row[1], row[2], row[3]
        if item_id not in sents:
            continue
        sentence_set = sents[item_id]
        if not sentence_set.sentences:
            continue
        sentence_obj = sentence_set.sentences[0]

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
                if len(tok0) != expected_first_len:
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
                if len(tok) != len(tgt_tok):
                    length_errors += 1
                    if len(examples) < args.max_examples:
                        examples.append((item_id, i, f"len-mismatch:{len(tok)}!={len(tgt_tok)}", tok))
                    continue

            try:
                lab = sentence_obj.labels[i]
                hidden = sentence_obj.hiddens[lab]
                target_s = sentence_obj.surprisal[lab]
                dist_s = model.get_surprisal_from_hidden(hidden, tok)
                delta = dist_s - target_s
                if delta <= args.min_delta:
                    plausible_errors += 1
                    if len(examples) < args.max_examples:
                        examples.append((item_id, i, f"delta={delta:.3f}", tok))
            except Exception:
                plausible_errors += 1
                if len(examples) < args.max_examples:
                    examples.append((item_id, i, "scoring-error", tok))

    print(f"positions_total={total_positions}")
    print(f"placeholder_errors={placeholder_errors}")
    print(f"nonword_errors={nonword_errors}")
    print(f"length_errors={length_errors}")
    print(f"plausible_or_bad_delta_errors={plausible_errors}")
    if total_positions > 0:
        rate = 100.0 * plausible_errors / total_positions
        print(f"plausible_error_rate_pct={rate:.2f}")

    if examples:
        print("examples:")
        for ex in examples:
            print(ex)

    failed = (placeholder_errors > 0) or (nonword_errors > 0) or (length_errors > 0) or (plausible_errors > 0)
    if args.strict and failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
