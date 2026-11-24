"""Hugging Face German causal-LM adapter for the Maze pipeline.

Shares all common logic with `scoring/hf_scorer.py` and overrides only what is
genuinely German-specific:

- reads the `HF_TOKEN` env var (gerpt2 mirrors may be gated).

`get_surprisal_from_hidden` used to be overridden here to score only the FIRST
BPE subtoken of a candidate, while the batch scorer that does the actual
selection summed all subtokens.  The two paths therefore returned different
numbers for the same word, and the single-word path systematically
under-reported the surprisal of any multi-token candidate -- which in German,
with its long compounds, is most of them.  The override is gone; German now
uses the shared exact multi-token sum.
"""

from maze_automate.scoring.hf_scorer import HFCausalScorer


class GermanScorer(HFCausalScorer):
    LANG_NAME = "German"
    DEFAULT_MODEL = "benjamin/gerpt2"
    USE_HF_TOKEN = True


__all__ = ["GermanScorer"]
