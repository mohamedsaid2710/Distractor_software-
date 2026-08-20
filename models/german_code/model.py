"""Hugging Face German causal-LM adapter for the Maze pipeline.

Shares all common logic with `models/hf_scorer.py` and overrides only what is
genuinely German-specific:

- reads the `HF_TOKEN` env var (gerpt2 mirrors may be gated);
- uses the legacy `batch_size` params key for its stored batch size
  (`sentence_set.py` passes `model_batch_size` explicitly per call).

`get_surprisal_from_hidden` used to be overridden here to score only the FIRST
BPE subtoken of a candidate, while the batch scorer that does the actual
selection summed all subtokens.  The two paths therefore returned different
numbers for the same word, and the single-word path systematically
under-reported the surprisal of any multi-token candidate -- which in German,
with its long compounds, is most of them.  The override is gone; German now
uses the shared exact multi-token sum.
"""

from models.hf_scorer import HFCausalScorer


class GermanScorer(HFCausalScorer):
    LANG_NAME = "German"
    DOWNLOAD_FLAG = "--german"
    DEFAULT_MODEL = "benjamin/gerpt2"
    USE_HF_TOKEN = True

    def _set_batch_size(self, params):
        # Legacy behavior: German reads `batch_size` (default 256), not
        # `model_batch_size`. Callers that want the bigger configured batches
        # pass `batch_size=` explicitly to get_surprisal_batch_from_hidden.
        self.batch_size = int(params.get("batch_size", 256))


__all__ = ["GermanScorer"]
