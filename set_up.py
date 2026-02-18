#!/usr/bin/env python3

import argparse
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser(description='Download and set-up files needed for Maze Automation. Specify which models to set-up')

parser.add_argument('--english', help="for English transformer model", action="store_true")

parser.add_argument('--german', help="for German model", action="store_true") 

args = parser.parse_args()
# need to check that all needed modules are installed
# download and place files

def download_english():
    check = check_pkgs(['torch', 'transformers'])
    make_dirs(['models/english_code', 'data/english_data'])
    if check:
        print("English setup: using Hugging Face model 'openai-community/gpt2-medium'.")
        print("The model will be auto-downloaded on first run (or loaded from local cache).")
    else:
        print("Some required packages are missing. Please install packages and try again.")
    return

def download_german():
    # require transformers + torch for the HF wrapper; if not installed we'll still create the wrapper file
    check = check_pkgs(['torch', 'transformers'])
    make_dirs(['models/german_code', 'data/german_data'])
    # If a local model wrapper already exists, don't overwrite it
    if not os.path.exists('models/german_code/model.py'):
        # Write a small Hugging Face wrapper so the set_up script does not attempt to wget a non-existent URL
        wrapper = r'''
import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

class GermanScorer:
    def __init__(self, model_name="dbmdz/german-gpt2", device=None,
                 use_wordfreq=False, zipf_threshold=3.0, freq_path=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_len = getattr(self.model.config, "n_positions",
                               getattr(self.model.config, "max_position_embeddings", 1024))
        self.use_wordfreq = use_wordfreq
        self.zipf_threshold = zipf_threshold
        self.freq_map = None
        if use_wordfreq and freq_path and os.path.exists(freq_path):
            try:
                with open(freq_path, "r", encoding="utf-8") as fh:
                    self.freq_map = json.load(fh)
            except Exception:
                self.freq_map = None

    def _candidate_zipf(self, candidate):
        if self.freq_map:
            vals = [self.freq_map.get(w, -10.0) for w in candidate.split()]
            return sum(vals) / max(1, len(vals))
        try:
            from wordfreq import zipf_frequency
        except Exception:
            return 0.0
        vals = [zipf_frequency(w, "de") for w in candidate.split()]
        return sum(vals) / max(1, len(vals))

    def score(self, context: str, candidate: str) -> float:
        if self.use_wordfreq:
            z = self._candidate_zipf(candidate)
            if z < self.zipf_threshold:
                return float("-inf")

        ctx_ids = self.tokenizer.encode(context, add_special_tokens=False)
        cand_ids = self.tokenizer.encode(candidate, add_special_tokens=False)
        if len(cand_ids) == 0:
            return float("-inf")

        allowed_ctx = max(0, self.max_len - len(cand_ids))
        if len(ctx_ids) > allowed_ctx:
            ctx_ids = ctx_ids[-allowed_ctx:]

        # Score context+candidate sequence and extract candidate token log-probs
        input_ids = torch.tensor([ctx_ids + cand_ids], device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)
            target_ids = input_ids[:, 1:]
            token_logps = log_probs[:, :-1, :].gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            cont_start = len(ctx_ids)
            cont_len = len(cand_ids)
            start_idx = cont_start - 1
            end_idx = start_idx + cont_len
            if start_idx < 0:
                # fallback: score candidate without context
                return self.score("", candidate)
            cont_logps = token_logps[0, start_idx:end_idx]
            return float(cont_logps.sum().item())
'''
        try:
            with open('models/german_code/model.py', 'w', encoding='utf-8') as fh:
                fh.write(wrapper)
            print("Wrote models/german_code/model.py (Hugging Face GermanScorer).")
        except Exception as e:
            print("Failed to write german_code/model.py:", e)

    # Do not attempt to download from placeholder URLs; instead instruct the user if they want a local checkpoint.
    # If you want to provide a local RNN checkpoint, place it under german_data/ and update your params_de.txt accordingly.
    print("German setup: wrapper created. Install 'transformers' and 'torch' (if not present) and the HF model will be auto-downloaded on first use.")
    print("If you prefer a local checkpoint/vocab, place german_code.py and into data/german_data/ and update the code.")
    return


def check_pkgs(packages):
    '''Given a list of packages, checks if they are installed
    If one is not installed, displays an error
    If any not installed, returns -1, else returns 1'''
    value=True
    for p in packages:
        try:
            __import__(p)
        except:
            print ("Needed package "+p+" is not installed.")
            value=False
    return value

def make_dirs(paths):
    '''Checks if paths exist, and if not creates them
    No return'''
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
    return
        

if args.english:
    download_english()
elif args.german:
    download_german()
check_pkgs(['wordfreq','nltk'])
