"""Hugging Face German causal-LM adapter for the Maze pipeline."""

import os
import math
import logging
import torch
import torch.nn.functional as F

# Keep Transformers on the PyTorch path only to avoid noisy TensorFlow init logs.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from transformers import AutoTokenizer, AutoModelForCausalLM
from lang_model import lang_model


class GermanScorer(lang_model):
    """Adapter that makes a Hugging Face causal LM implement `lang_model`."""

    def __init__(self, params=None):
        params = params or {}
        model_name_param = params.get("hf_model_name", "models/dbmdz-german-gpt2")

        if not os.path.isabs(model_name_param) and not os.path.exists(model_name_param):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            candidate = os.path.join(base_dir, model_name_param)
            model_name = candidate if os.path.exists(candidate) else model_name_param
        else:
            model_name = model_name_param

        device = params.get("device", None)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = self._load_tokenizer(model_name)
        self.model = self._load_model(model_name).to(self.device)
        self.model.eval()

        self.max_len = getattr(
            self.model.config,
            "n_positions",
            getattr(self.model.config, "max_position_embeddings", 1024),
        )

        # GPT-2 style models may have no pad token by default; align it to eos.
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
        # Prefer local cache/files first.
        try:
            return AutoTokenizer.from_pretrained(model_name, use_fast=False, local_files_only=True)
        except Exception:
            pass
        try:
            return AutoTokenizer.from_pretrained(model_name, use_fast=False)
        except Exception as e:
            raise RuntimeError(
                "Failed to load German tokenizer '%s'. "
                "If this machine is offline, run `python download_model.py --german` "
                "and set `hf_model_name` to the local model path."
                % model_name
            ) from e

    def _load_model(self, model_name):
        # If user points to a local directory, fail fast with a clear message.
        if self._is_local_dir(model_name) and not self._has_local_weights(model_name):
            raise RuntimeError(
                "Local model directory '%s' does not contain model weights "
                "(expected e.g. `pytorch_model.bin` or `model.safetensors`). "
                "Download with `python download_model.py --german`."
                % model_name
            )
        # First try strict local mode.
        try:
            return AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
        except Exception:
            pass
        # Then allow remote lookup.
        try:
            return AutoModelForCausalLM.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(
                "Failed to load German model '%s'. "
                "If this machine is offline, run `python download_model.py --german` "
                "and set `hf_model_name` to the local model path."
                % model_name
            ) from e

    def tokenize(self, word):
        return self.tokenizer.tokenize(word)

    def empty_sentence(self):
        return []

    def update(self, hidden, word):
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


__all__ = ["GermanScorer"]
