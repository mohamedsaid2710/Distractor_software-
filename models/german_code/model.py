"""Hugging Face German causal-LM adapter for the Maze pipeline.

Shares all common logic with `models/hf_scorer.py` and overrides only what is
genuinely German-specific:

- reads the `HF_TOKEN` env var (gerpt2 mirrors may be gated);
- uses the legacy `batch_size` params key for its stored batch size
  (`sentence_set.py` passes `model_batch_size` explicitly per call);
- `get_surprisal_from_hidden` scores only the FIRST subtoken of a candidate
  from the next-word distribution, instead of the exact multi-token sum.
  NOTE: this makes the German single-word path deliberately cheaper but
  inconsistent with the batch scorer, which sums all subtokens. Kept as-is to
  preserve validated German generation behavior.
"""

import torch
import torch.nn.functional as F

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

    def get_surprisal_from_hidden(self, hidden, word):
        """Surprisal of `word`'s FIRST subtoken given context (hidden = token IDs)."""
        ctx_ids = list(hidden) if isinstance(hidden, (list, tuple)) else list(hidden)
        if not ctx_ids:
            # No context - use the batch scorer's internal handling
            return self.get_surprisal_batch_from_hidden([], [word], batch_size=1)[0]

        allowed_ctx = self.context_limit(1)
        ctx = ctx_ids[-allowed_ctx:] if len(ctx_ids) > allowed_ctx else ctx_ids

        input_ids = torch.tensor([ctx], device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1).clamp(min=1e-12)
            surprisals = -torch.log2(probs)

        parts = self.encode_word(word)
        if len(parts) == 0:
            return 0.0
        token = parts[0]
        if token >= surprisals.size(0):
            return 0.0
        return float(surprisals[token].item())


__all__ = ["GermanScorer"]
