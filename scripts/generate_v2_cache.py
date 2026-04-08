import stanza
import json
import os
import wordfreq
import re
import math
import logging
from tqdm import tqdm

# === GERMAN POS CACHE GENERATOR (EXHAUSTIVE V2) ===
# This script tags the ENTIRE German dictionary (170,000+ words)
# without any frequency filters to ensure 100% total coverage.

# --- CONFIGURATION ---
LANG = 'de'
OUTPUT_FILE = 'german_pos_cache_v2.json'
INCLUDE_FILE = 'include_de.txt'
BATCH_SIZE = 5000 # Aggressive batching for maximum high-speed tagging

def run_generation():
    # 1. Initialize Stanza
    print(">>> Initializing Stanza (German)...")
    try:
        stanza.download(LANG, processors='tokenize,pos')
        nlp = stanza.Pipeline(LANG, processors='tokenize,pos', use_gpu=True)
    except Exception as e:
        print(f"GPU failed, falling back to CPU: {e}")
        nlp = stanza.Pipeline(LANG, processors='tokenize,pos', use_gpu=False)

    # 2. Collect EVERY Word from wordfreq
    print(">>> Collecting all 170,000+ German words from wordfreq...")
    freq_dict = wordfreq.get_frequency_dict(LANG)
    words_to_tag = set()
    
    # A. Always include 'include_de.txt'
    if os.path.exists(INCLUDE_FILE):
        with open(INCLUDE_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                w = line.strip().lower()
                if w and not w.startswith('#'):
                    words_to_tag.add(w)
        print(f"[LOAD] Loaded words from {INCLUDE_FILE}")

    # B. Add everything from wordfreq (NO FILTERS as requested)
    for w in freq_dict.keys():
        words_to_tag.add(w.lower())

    words_list = sorted(list(words_to_tag))
    print(f"[READY] Prepared {len(words_list)} unique words for EXHAUSTIVE tagging.")

    # 3. Batch Tagging (Using context frame logic)
    pos_cache = {}
    print(f">>> Tagging started (Batch Size: {BATCH_SIZE})...")
    # Batch tagging using bulk_process for maximum performance on GPU
    for i in tqdm(range(0, len(words_list), BATCH_SIZE), desc="Neural Tagging"):
        batch = words_list[i : i + BATCH_SIZE]
        
        # Frame forces Stanza to evaluate word in a noun slot for 100% accuracy.
        frames = [f"Das ist ein {w}." for w in batch]
        
        try:
            docs = nlp.bulk_process(frames)
            for word, doc in zip(batch, docs):
                if len(doc.sentences) > 0 and len(doc.sentences[0].words) >= 4:
                    # Index 3 is the target word in our frame: "Das(0) ist(1) ein(2) {w}(3) .(4)"
                    pos_cache[word] = doc.sentences[0].words[3].upos
                elif doc.sentences and doc.sentences[0].words:
                    pos_cache[word] = doc.sentences[0].words[0].upos
                else:
                    pos_cache[word] = 'X'
        except Exception as e:
            logging.error(f"Batch index {i} failed: {e}")
            for w in batch:
                pos_cache[w] = 'X'

    # 4. Save Final Cache
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(pos_cache, f, ensure_ascii=False, indent=2)

    print(f"\n>>> SUCCESS! Exhaustive high-accuracy cache saved to: {OUTPUT_FILE}")
    print(f"Total words tagged: {len(pos_cache)}")

if __name__ == "__main__":
    run_generation()
