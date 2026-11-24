"""Hugging Face English causal-LM adapter for the Maze pipeline.

All scoring logic lives in `scoring/hf_scorer.py`; this subclass only sets the
English defaults.
"""

from maze_automate.scoring.hf_scorer import HFCausalScorer


class EnglishScorer(HFCausalScorer):
    LANG_NAME = "English"
    DEFAULT_MODEL = "openai-community/gpt2-medium"


__all__ = ["EnglishScorer"]
