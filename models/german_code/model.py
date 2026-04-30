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
        model_name_param = params.get("hf_model_name", "benjamin/gerpt2")

        if not os.path.isabs(model_name_param) and not os.path.exists(model_name_param):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            candidate = os.path.join(base_dir, model_name_param)
            model_name = candidate if os.path.exists(candidate) else model_name_param
        else:
            model_name = model_name_param

        device = params.get("device", None)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f">>> [SCORER] Initializing German Scorer with model: {model_name}")
        
        # Store batch_size from params (default 256 for backward compatibility)
        self.batch_size = int(params.get("batch_size", 256))

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
        token = os.environ.get("HF_TOKEN", None)
        # Prefer local cache/files first, then allow online lookup.
        # Try slow tokenizer first for backwards compatibility, then fast tokenizer
        # so partial caches containing only tokenizer.json still work offline.
        attempts = (
            {"use_fast": False, "local_files_only": True, "token": token},
            {"use_fast": True, "local_files_only": True, "token": token},
            {"use_fast": False, "token": token},
            {"use_fast": True, "token": token},
        )
        last_error = None
        for kwargs in attempts:
            try:
                return AutoTokenizer.from_pretrained(model_name, **kwargs)
            except Exception as e:
                last_error = e
        raise RuntimeError(
            "Failed to load German tokenizer '%s'. "
            "If this machine is offline, run `python download_model.py --german` "
            "and set `hf_model_name` to the local model path."
            % model_name
        ) from last_error

    def _load_model(self, model_name):
        token = os.environ.get("HF_TOKEN", None)
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
            return AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, token=token)
        except Exception:
            pass
        # Then allow remote lookup.
        try:
            return AutoModelForCausalLM.from_pretrained(model_name, token=token)
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
        """Compute surprisal of a word given context (hidden state = token IDs)."""
        ctx_ids = list(hidden) if isinstance(hidden, (list, tuple)) else list(hidden)
        if not ctx_ids:
            # No context - use model's internal handling
            return self.get_surprisal_batch_from_hidden([], [word], batch_size=1)[0]
        
        allowed_ctx = max(0, self.max_len - 1)
        ctx = ctx_ids[-allowed_ctx:] if len(ctx_ids) > allowed_ctx else ctx_ids
        
        input_ids = torch.tensor([ctx], device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1).clamp(min=1e-12)
            surprisals = -torch.log2(probs)
        
        parts = self.tokenizer.encode(word, add_special_tokens=False)
        if len(parts) == 0:
            return 0.0
        token = parts[0]
        if token >= surprisals.size(0):
            return 0.0
        return float(surprisals[token].item())

    def get_surprisal_batch_from_hidden(self, hidden, words, batch_size=None):
        """Score a list of words in parallel batches using Multi-Token Summation.
        
        This aligns identically with get_surprisal_from_hidden by scoring and 
        summing the joint log-probability of ALL sub-tokens for a given word.
        
        Args:
            hidden: Context token IDs
            words: List of words to score
            batch_size: Override instance batch_size (default uses self.batch_size from params)
        """
        if not words:
            return []
        
        if batch_size is None:
            batch_size = getattr(self, 'model_batch_size', getattr(self, 'batch_size', 500))
            
        ctx_ids = list(hidden) if isinstance(hidden, (list, tuple)) else list(hidden)
        ctx_len = len(ctx_ids)
        all_results = []
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        # Focused Context window
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
                
                # Get log probabilities over the entire vocabulary for the sequence
                log_probs = F.log_softmax(outputs.logits, dim=-1)
                
                # We need the log-probs of the true target tokens (shift right)
                # input_tensor[:, 1:] are the actual tokens observed
                target_ids = input_tensor[:, 1:]
                # Gather log-probs for the specific observed tokens
                token_logps = log_probs[:, :-1, :].gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
                
                for b_idx in range(len(chunk)):
                    c_len = word_lengths[b_idx]
                    
                    start_idx = n_ctx - 1
                    end_idx = start_idx + c_len
                    
                    if start_idx < 0:
                        start_idx = 0
                        end_idx = c_len
                    
                    # Prevent out-of-bounds just in case
                    end_idx = min(end_idx, token_logps.shape[1])
                    
                    # Slice exact sub-tokens for the word
                    selected = token_logps[b_idx, start_idx:end_idx]
                    
                    # Sum log-probabilities to compute joint probability for the target word
                    total_ln = -selected.sum().item()
                    all_results.append(float(total_ln / math.log(2)))
                
                # Cleanup
                del outputs
                del log_probs
                del token_logps
                if i % (batch_size * 2) == 0:
                    torch.cuda.empty_cache()

        return all_results


__all__ = ["GermanScorer"]
