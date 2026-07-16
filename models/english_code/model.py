"""Hugging Face English causal-LM adapter for the Maze pipeline.

All scoring logic lives in `models/hf_scorer.py`; this subclass only sets the
English defaults.
"""

from models.hf_scorer import HFCausalScorer


class EnglishScorer(HFCausalScorer):
    LANG_NAME = "English"
    DOWNLOAD_FLAG = "--english"
    DEFAULT_MODEL = "openai-community/gpt2-medium"


__all__ = ["EnglishScorer"]
