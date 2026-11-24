"""Targets and candidates are scored the same way: all subtokens summed.

Uses a tiny randomly initialised GPT-2 and a toy tokenizer, so no download is
needed.  The toy tokenizer splits every word into one token per letter, which
makes every word multi-token.
"""
import math

import pytest
import torch

transformers = pytest.importorskip("transformers")

from maze_automate.scoring.hf_scorer import HFCausalScorer  # noqa: E402


class LetterTokenizer:
    """' ab' -> [SPACE, a, b]-style ids: id 1 = space marker, letters 2..27."""
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        ids = []
        for i, ch in enumerate(text):
            if ch == " ":
                continue
            # the space before a word is folded into its first letter, like BPE's "Ġ"
            space = i > 0 and text[i - 1] == " "
            ids.append((28 if space else 2) + (ord(ch.lower()) - ord("a")) % 26)
        return ids


@pytest.fixture(scope="module")
def scorer():
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=60, n_positions=64, n_embd=32,
                                  n_layer=2, n_head=2)
    s = HFCausalScorer.__new__(HFCausalScorer)
    s.model = transformers.GPT2LMHeadModel(cfg).eval()
    s.tokenizer = LetterTokenizer()
    s._prefix_space = True
    s.device = "cpu"
    s.max_len = 64
    s.context_window = None
    s.model_batch_size = 8
    return s


def _context(scorer, words):
    hidden = []
    for w in words:
        hidden, _ = scorer.update(hidden, w)
    return hidden


def _chain_rule(scorer, hidden, word):
    """Surprisal by hand: one model call per subtoken, -log2 p summed."""
    ids = list(hidden)
    total = 0.0
    for tok in scorer.encode_word(word, at_start=not hidden):
        with torch.no_grad():
            logits = scorer.model(torch.tensor([ids])).logits[0, -1]
        total += -torch.log_softmax(logits, -1)[tok].item() / math.log(2)
        ids.append(tok)
    return total


def test_single_word_score_is_the_sum_over_subtokens(scorer):
    hidden = _context(scorer, ["the", "cat"])
    assert len(scorer.encode_word("kitchen")) == 7          # multi-token
    assert scorer.get_surprisal_from_hidden(hidden, "kitchen") == \
        pytest.approx(_chain_rule(scorer, hidden, "kitchen"), abs=1e-4)


def test_batch_and_single_give_the_same_number(scorer):
    """Candidates are scored in batches, targets one at a time."""
    hidden = _context(scorer, ["the", "cat", "sat"])
    words = ["on", "kitchen", "a", "elephant", "ran"]       # mixed lengths -> padding
    batch = scorer.get_surprisal_batch_from_hidden(hidden, words, batch_size=3)
    single = [scorer.get_surprisal_from_hidden(hidden, w) for w in words]
    assert batch == pytest.approx(single, abs=1e-4)


def test_target_is_scored_without_its_punctuation():
    """`Sentence.do_surprisal` strips punctuation the way candidates are bare."""
    from maze_automate.selection import Sentence

    calls = []

    class Spy:
        def get_surprisal_from_hidden(self, hidden, word):
            calls.append(word)
            return 0.0

    s = Sentence(["The", "cat", "sat."], ["0", "1", "2"], "1", "t", "The cat sat.")
    s.hiddens = {"1": [1], "2": [1, 2]}
    s.do_surprisal(Spy())
    assert calls == ["cat", "sat"]
