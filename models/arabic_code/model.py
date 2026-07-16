"""Hugging Face Arabic causal-LM adapter for the Maze pipeline.

All scoring logic lives in `models/hf_scorer.py`; this subclass only sets the
Arabic defaults. (Diacritic stripping happens in the dictionary layer, not here:
the scorer receives already-normalized words.)
"""

from models.hf_scorer import HFCausalScorer


class ArabicScorer(HFCausalScorer):
    LANG_NAME = "Arabic"
    DOWNLOAD_FLAG = "--arabic"
    DEFAULT_MODEL = "aubmindlab/aragpt2-medium"


__all__ = ["ArabicScorer"]
