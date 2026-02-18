
"""Hugging Face German causal-LM .

This module provides `GermanScorer` which implements the repo's `lang_model` API
(tokenize, empty_sentence, update, get_surprisal) and can be loaded by the pipeline via `model_loc: "german_code"` and `model_class: "GermanScorer"` in params.
"""
import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from lang_model import lang_model
import math
import logging


class GermanScorer(lang_model):
	"""
	Adapter making a Hugging Face causal LM behave like the repo's `lang_model`.

	Implements `tokenize`, `empty_sentence`, `update`, and `get_surprisal` so the model can be plugged into the existing pipeline without changing `main.py` or `sentence_set.py`.
	"""

	def __init__(self, params=None):
		params = params or {}
		model_name_param = params.get("hf_model_name", "models/dbmdz-german-gpt2")
		
		# Robust path resolution: Handle both relative and absolute paths
		# If the relative path doesn't exist from CWD, try finding it relative to this file's package root
		if not os.path.isabs(model_name_param) and not os.path.exists(model_name_param):
			# .../models/german_code/model.py -> .../ (package root)
			base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
			candidate = os.path.join(base_dir, model_name_param)
			if os.path.exists(candidate):
				model_name = candidate
			else:
				model_name = model_name_param # Fallback to original, might fail
		else:
			model_name = model_name_param

		device = params.get("device", None)
		use_wordfreq = params.get("use_wordfreq", False)
		freq_path = params.get("include_words", None)

		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		# load tokenizer and model
		# We set local_files_only=True for tokenizer too, to prevent accidental Hugging Face lookups if path is wrong
		try:
			self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, local_files_only=True)
		except OSError:
			# Fallback: try without local_files_only if it's actually a valid hub reference (rare case here)
			self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
			
		self.model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True).to(self.device)
		self.model.eval()
		# maximum context length
		self.max_len = getattr(self.model.config, "n_positions",
						getattr(self.model.config, "max_position_embeddings", 1024))

		self.use_wordfreq = use_wordfreq
		self.freq_map = None
		if self.use_wordfreq and freq_path and os.path.exists(freq_path):
			try:
				with open(freq_path, "r", encoding="utf-8") as fh:
					self.freq_map = json.load(fh)
			except Exception:
				self.freq_map = None

		# Ensure embedding matrix matches tokenizer size (if special tokens were added)
		try:
			tok_len = len(self.tokenizer)
			emb = self.model.get_input_embeddings()
			emb_rows = emb.weight.shape[0]
			if tok_len != emb_rows:
				logging.info("Resizing model embeddings: %d -> %d", emb_rows, tok_len)
				self.model.resize_token_embeddings(tok_len)
				# initialize new embeddings to mean of existing (avoid pure-random vectors)
				with torch.no_grad():
					emb = self.model.get_input_embeddings()
					if tok_len > emb_rows:
						mean_emb = emb.weight[:emb_rows].mean(dim=0)
						for idx in range(emb_rows, tok_len):
							emb.weight[idx].copy_(mean_emb)
		except Exception as e:
			logging.debug("Embedding resize/init skipped or failed: %s", e)

	def tokenize(self, word):
		"""Return tokenizer-level tokens for a word (strings)."""
		return self.tokenizer.tokenize(word)

	def empty_sentence(self):
		"""Return an empty hidden/context representation (list of token ids)."""
		return []

	def update(self, hidden, word):
		"""Append `word` to hidden (list of token ids) and return (new_hidden, surprisals_tensor).

		`surprisals_tensor` is a 1-D torch tensor over model vocab with surprisal in bits.
		"""
		# token ids for the word (no special tokens)
		parts = self.tokenizer.encode(word, add_special_tokens=False)
		if not isinstance(hidden, list):
			# defensive: if hidden was None or a tensor, convert to list
			try:
				hidden = list(hidden)
			except Exception:
				hidden = []
		new_hidden = hidden + parts

		# truncate if longer than model capacity
		allowed_ctx = max(0, self.max_len - 1)  # reserve at least one slot
		if len(new_hidden) > allowed_ctx:
			ctx = new_hidden[-allowed_ctx:]
		else:
			ctx = new_hidden

		input_ids = torch.tensor([ctx], device=self.device)
		with torch.no_grad():
			outputs = self.model(input_ids)
			logits = outputs.logits  # shape (1, seq_len, vocab)
			last_logits = logits[0, -1, :]
			probs = F.softmax(last_logits, dim=-1)
			# convert to surprisal in bits, avoid zeros
			probs = probs.clamp(min=1e-12)
			surprisals = -torch.log2(probs)

		return new_hidden, surprisals

	def get_surprisal(self, surprisals, word):
		"""Return surprisal (bits) for the first subtoken of `word` using `surprisals` tensor."""
		parts = self.tokenizer.encode(word, add_special_tokens=False)
		if len(parts) == 0:
			return 0
		token = parts[0]
		if token >= surprisals.size(0):
			# token id out of range for surprisal vector (shouldn't happen)
			return 0
		if len(parts) > 1:
			# warn: multi-subtoken word — we return the surprisal of the first subtoken
			logging.info('Word %s is multi-token; using first subtoken for surprisal.', word)
		return float(surprisals[token].item())

	def get_surprisal_from_hidden(self, hidden, word):
		"""Compute full-word surprisal (bits) for `word` given `hidden` context.

		This method builds the input sequence as hidden + candidate tokens, runs the model once,
		and extracts the log-probabilities for the candidate tokens conditioned on the context.
		Returns surprisal in bits (sum across subtokens).
		"""
		parts = self.tokenizer.encode(word, add_special_tokens=False)
		if len(parts) == 0:
			return 0.0

		# prepare context token ids as list
		ctx = list(hidden) if isinstance(hidden, (list, tuple)) else list(hidden)
		# truncate context if needed
		allowed_ctx = max(0, self.max_len - len(parts))
		if len(ctx) > allowed_ctx:
			ctx = ctx[-allowed_ctx:]

		input_ids = torch.tensor([ctx + parts], device=self.device)
		with torch.no_grad():
			outputs = self.model(input_ids)
			logits = outputs.logits
			log_probs = F.log_softmax(logits, dim=-1)  # natural log
			target_ids = input_ids[:, 1:]
			token_logps = log_probs[:, :-1, :].gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
			cont_start = len(ctx)
			cont_len = len(parts)
			start_idx = cont_start - 1
			end_idx = start_idx + cont_len
			if start_idx < 0:
				# fallback: score candidate without context
				# compute from token_logps when no context: treat first token prediction differently
				start_idx = 0
				end_idx = cont_len
			# token_logps are natural log probabilities; convert to bits
			selected = token_logps[0, start_idx:end_idx]
			# sum of -log2(p) = -sum(ln p)/ln2
			total_ln = -selected.sum().item()
			total_bits = total_ln / math.log(2)
			return float(total_bits)


__all__ = ["GermanScorer"]
