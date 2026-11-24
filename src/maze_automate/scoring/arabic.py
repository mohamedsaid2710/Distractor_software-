"""Hugging Face Arabic causal-LM adapter for the Maze pipeline.

All scoring logic lives in `scoring/hf_scorer.py`; this subclass only sets the
Arabic defaults. (Diacritic stripping happens in the dictionary layer, not here:
the scorer receives already-normalized words.)
"""

from maze_automate.scoring.hf_scorer import HFCausalScorer


class ArabicScorer(HFCausalScorer):
    LANG_NAME = "Arabic"
    DEFAULT_MODEL = "aubmindlab/aragpt2-medium"


__all__ = ["ArabicScorer"]
