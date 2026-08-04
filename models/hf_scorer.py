"""Shared Hugging Face causal-LM scorer for the Maze pipeline.

All three language adapters (English, German, Arabic) are thin subclasses of
`HFCausalScorer` below. A subclass only declares what actually differs:

- `LANG_NAME`       — used in log and error messages ("English", "German", ...)
- `DOWNLOAD_FLAG`   — the `download_model.py` flag suggested in offline errors
- `DEFAULT_MODEL`   — Hugging Face model ID used when params omit `hf_model_name`
- `USE_HF_TOKEN`    — whether to read the `HF_TOKEN` env var for gated repos
- method overrides  — e.g. German overrides `get_surprisal_from_hidden`

Scoring model: a "hidden state" in this pipeline is simply the list of context
token IDs seen so far (not a neural hidden state). Surprisal is measured in
bits: -log2 P(word | context). For multi-token (BPE-split) words the batch
scorer sums the conditional log-probabilities of ALL subtokens, which is the
exact joint surprisal of the word.
"""

import os
import math
import logging
import torch
import torch.nn.functional as F

# Keep Transformers on the PyTorch path only to avoid noisy TensorFlow init logs.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from transformers import AutoTokenizer, AutoModelForCausalLM
from lang_model import lang_model


class HFCausalScorer(lang_model):
    """Adapter that makes a Hugging Face causal LM implement `lang_model`."""

    LANG_NAME = "English"
    DOWNLOAD_FLAG = "--english"
    DEFAULT_MODEL = "openai-community/gpt2-medium"
    USE_HF_TOKEN = False

    def __init__(self, params=None):
        params = params or {}
        model_name_param = params.get("hf_model_name", self.DEFAULT_MODEL)

        # Allow `hf_model_name` to be a path relative to the repo root.
        if not os.path.isabs(model_name_param) and not os.path.exists(model_name_param):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidate = os.path.join(base_dir, model_name_param)
            model_name = candidate if os.path.exists(candidate) else model_name_param
        else:
            model_name = model_name_param

        self.device = self._resolve_device(params.get("device", None))
        print(f">>> [SCORER] Initializing {self.LANG_NAME} Scorer with model: {model_name}")

        self._set_batch_size(params)

        self.tokenizer = self._load_tokenizer(model_name)
        self.model = self._load_model(model_name).to(self.device)
        self.model.eval()

        self.max_len = getattr(
            self.model.config,
            "n_positions",
            getattr(self.model.config, "max_position_embeddings", 1024),
        )

        # GPT-2 style models have no pad token by default; align it to eos for stable batching.
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            tok_len = len(self.tokenizer)
            emb = self.model.get_input_embeddings()
            emb_rows = emb.weight.shape[0]
            if tok_len != emb_rows:
                logging.info("Resizing model embeddings: %d -> %d", emb_rows, tok_len)
                self.model.resize_token_embeddings(tok_len)
        except Exception as e:
            logging.debug("Embedding resize skipped or failed: %s", e)

    @staticmethod
    def _resolve_device(device):
        """Pick a torch device, falling back to CPU when the requested one is unusable.

        An explicit `device` in a params file used to be taken at face value, so a
        config written on a CUDA machine crashed on CPU-only hosts (VMs, laptops
        without an NVIDIA driver). Now an unavailable device only costs a warning.
        """
        requested = str(device or "").lower()

        if not requested:
            return "cuda" if torch.cuda.is_available() else "cpu"

        if requested.startswith("cuda") and not torch.cuda.is_available():
            logging.warning("CUDA requested but unavailable on this machine; falling back to CPU.")
            return "cpu"

        try:
            # Cheap probe: allocating one element surfaces missing drivers/backends here
            # rather than midway through `.to(...)` on the full model.
            torch.zeros(1, device=requested)
        except Exception as e:
            # Torch backend errors can run to hundreds of lines (the full dispatch
            # table); the first line carries the actual reason.
            reason = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            logging.warning("Device %r unusable (%s); falling back to CPU.", requested, reason[:200])
            return "cpu"

        return requested

    def _set_batch_size(self, params):
        """Store the GPU scoring batch size (subclasses may use a legacy key)."""
        self.model_batch_size = int(params.get("model_batch_size", params.get("batch_size", 256)))

    def _hf_token(self):
        """Auth token for gated Hugging Face repos (only when USE_HF_TOKEN is set)."""
        return os.environ.get("HF_TOKEN", None) if self.USE_HF_TOKEN else None

    @staticmethod
    def _is_local_dir(model_name):
        return os.path.isdir(model_name)

    @staticmethod
    def _has_local_weights(model_dir):
        """Return True if a local HF model directory has a recognizable weight file."""
        weight_markers = (
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.bin.index.json",
            "model.safetensors.index.json",
        )
        return any(os.path.exists(os.path.join(model_dir, m)) for m in weight_markers)

    def _load_tokenizer(self, model_name):
        # Prefer local cache/files first, then allow online lookup.
        # Try slow tokenizer first for backwards compatibility, then fast tokenizer
        # so partial caches containing only tokenizer.json still work offline.
        attempts = (
            {"use_fast": False, "local_files_only": True},
            {"use_fast": True, "local_files_only": True},
            {"use_fast": False},
            {"use_fast": True},
        )
        if self.USE_HF_TOKEN:
            token = self._hf_token()
            attempts = tuple({**kwargs, "token": token} for kwargs in attempts)
        last_error = None
        for kwargs in attempts:
            try:
                return AutoTokenizer.from_pretrained(model_name, **kwargs)
            except Exception as e:
                last_error = e
        raise RuntimeError(
            "Failed to load %s tokenizer '%s'. "
            "If this machine is offline, run `python3 download_model.py %s` "
            "and set `hf_model_name` to the local model path."
            % (self.LANG_NAME, model_name, self.DOWNLOAD_FLAG)
        ) from last_error

    def _load_model(self, model_name):
        # If user points to a local directory, fail fast with a clear message.
        if self._is_local_dir(model_name) and not self._has_local_weights(model_name):
            raise RuntimeError(
                "Local model directory '%s' does not contain model weights "
                "(expected e.g. `pytorch_model.bin` or `model.safetensors`). "
                "Download with `python3 download_model.py %s`."
                % (model_name, self.DOWNLOAD_FLAG)
            )
        token_kwargs = {"token": self._hf_token()} if self.USE_HF_TOKEN else {}
        # First try strict local mode.
        try:
            return AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, **token_kwargs)
        except Exception:
            pass
        # Then allow remote lookup.
        try:
            return AutoModelForCausalLM.from_pretrained(model_name, **token_kwargs)
        except Exception as e:
            raise RuntimeError(
                "Failed to load %s model '%s'. "
                "If this machine is offline, run `python3 download_model.py %s` "
                "and set `hf_model_name` to the local model path."
                % (self.LANG_NAME, model_name, self.DOWNLOAD_FLAG)
            ) from e

    def tokenize(self, word):
        return self.tokenizer.tokenize(word)

    def empty_sentence(self):
        """A fresh context: no tokens seen yet."""
        return []

    def update(self, hidden, word):
        """Append `word` to the context and return (new context, next-word surprisals).

        The returned surprisal tensor covers the whole vocabulary and is what
        `get_surprisal` indexes into.
        """
        parts = self.tokenizer.encode(word, add_special_tokens=False)
        if not isinstance(hidden, list):
            try:
                hidden = list(hidden)
            except Exception:
                hidden = []
        new_hidden = hidden + parts

        allowed_ctx = max(0, self.max_len - 1)
        ctx = new_hidden[-allowed_ctx:] if len(new_hidden) > allowed_ctx else new_hidden

        input_ids = torch.tensor([ctx], device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1).clamp(min=1e-12)
            surprisals = -torch.log2(probs)
        return new_hidden, surprisals

    def get_surprisal(self, surprisals, word):
        """Surprisal of `word` from a precomputed next-word distribution.

        Only the FIRST subtoken is used here (fast path); the exact multi-token
        score lives in `get_surprisal_from_hidden` / the batch scorer.
        """
        parts = self.tokenizer.encode(word, add_special_tokens=False)
        if len(parts) == 0:
            return 0.0
        token = parts[0]
        if token >= surprisals.size(0):
            return 0.0
        if len(parts) > 1:
            logging.info("Word %s is multi-token; using first subtoken for surprisal.", word)
        return float(surprisals[token].item())

    def get_surprisal_from_hidden(self, hidden, word):
        """Exact surprisal of `word` after context `hidden` (sums ALL subtokens)."""
        parts = self.tokenizer.encode(word, add_special_tokens=False)
        if len(parts) == 0:
            return 0.0

        ctx = list(hidden) if isinstance(hidden, (list, tuple)) else list(hidden)
        allowed_ctx = max(0, self.max_len - len(parts))
        if len(ctx) > allowed_ctx:
            ctx = ctx[-allowed_ctx:]

        input_ids = torch.tensor([ctx + parts], device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)
            target_ids = input_ids[:, 1:]
            token_logps = log_probs[:, :-1, :].gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

            cont_start = len(ctx)
            cont_len = len(parts)
            start_idx = cont_start - 1
            end_idx = start_idx + cont_len
            if start_idx < 0:
                start_idx = 0
                end_idx = cont_len
            selected = token_logps[0, start_idx:end_idx]
            total_ln = -selected.sum().item()
            return float(total_ln / math.log(2))

    def get_surprisal_batch_from_hidden(self, hidden, words, batch_size=None):
        """Score a list of words in parallel batches using Multi-Token Summation.

        This aligns identically with `get_surprisal_from_hidden` by scoring and
        summing the joint log-probability of ALL sub-tokens for a given word.

        Args:
            hidden: Context token IDs.
            words: List of words to score.
            batch_size: Override the instance batch size for this call.
        """
        if not words:
            return []

        if batch_size is None:
            batch_size = getattr(self, 'model_batch_size', getattr(self, 'batch_size', 500))

        ctx_ids = list(hidden) if isinstance(hidden, (list, tuple)) else list(hidden)
        ctx_len = len(ctx_ids)
        all_results = []
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        # Focused context window: only the last 64 context tokens are used, which
        # keeps long sentences fast without materially changing local surprisal.
        MAX_CONTEXT = 64
        allowed_ctx = min(ctx_len, MAX_CONTEXT)
        active_ctx = ctx_ids[-allowed_ctx:] if ctx_len > 0 else []
        n_ctx = len(active_ctx)

        for i in range(0, len(words), batch_size):
            chunk = words[i:i + batch_size]
            batch_ids = []
            batch_masks = []
            word_lengths = []

            for w in chunk:
                parts = self.tokenizer.encode(w, add_special_tokens=False)
                if not parts:
                    parts = [pad_id]
                word_lengths.append(len(parts))
                full_seq = active_ctx + parts
                batch_ids.append(full_seq)
                batch_masks.append([1] * len(full_seq))

            # Right-pad every sequence in the chunk to the same length.
            max_batch_len = max(len(s) for s in batch_ids)
            padded_ids = []
            padded_masks = []
            for s, m in zip(batch_ids, batch_masks):
                diff = max_batch_len - len(s)
                padded_ids.append(s + [pad_id] * diff)
                padded_masks.append(m + [0] * diff)

            input_tensor = torch.tensor(padded_ids, device=self.device)
            mask_tensor = torch.tensor(padded_masks, device=self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor, attention_mask=mask_tensor)

                # Log-probabilities of each observed token given its prefix
                # (shift-by-one: logits at position t predict token t+1).
                log_probs = F.log_softmax(outputs.logits, dim=-1)
                target_ids = input_tensor[:, 1:]
                token_logps = log_probs[:, :-1, :].gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

                for b_idx in range(len(chunk)):
                    c_len = word_lengths[b_idx]

                    # Slice exactly the subtoken positions belonging to this word.
                    start_idx = n_ctx - 1
                    end_idx = start_idx + c_len
                    if start_idx < 0:
                        start_idx = 0
                        end_idx = c_len
                    end_idx = min(end_idx, token_logps.shape[1])
                    selected = token_logps[b_idx, start_idx:end_idx]

                    # Sum log-probabilities => joint surprisal of the word, in bits.
                    total_ln = -selected.sum().item()
                    all_results.append(float(total_ln / math.log(2)))

                # Free GPU memory between chunks.
                del outputs
                del log_probs
                del token_logps
                if i % (batch_size * 2) == 0:
                    torch.cuda.empty_cache()

        return all_results


__all__ = ["HFCausalScorer"]
