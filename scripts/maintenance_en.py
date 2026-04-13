import json
import os
import sys

# Ensure this is run from root or scripts dir
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.append(WORKSPACE_ROOT)

CACHE_FILE = os.path.join(WORKSPACE_ROOT, "models", "english_code", "english_pos_cache.json")

def run_maintenance():
    try:
        import spacy
    except ImportError:
        print("Please install spacy: pip install spacy")
        return
        
    print("Loading SpaCy en_core_web_lg...")
    try:
        nlp = spacy.load("en_core_web_lg")
    except:
        print("Downloading en_core_web_lg...")
        spacy.cli.download("en_core_web_lg")
        nlp = spacy.load("en_core_web_lg")
        
    print(f"Loading English cache from {CACHE_FILE}")
    if not os.path.exists(CACHE_FILE):
        print("Cache file not found!")
        return
        
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
        
    print(f"Processing {len(cache)} entries. This will enforce strict SpaCy token alignments...")
    words = list(cache.keys())
    corrected = {}
    changed = 0
    
    docs = list(nlp.pipe(words))
    for w, doc in zip(words, docs):
        old_pos = cache[w]
        pos = doc[0].pos_ if len(doc) > 0 else 'X'
        
        # Simplify tag
        if pos not in ('NOUN', 'ADJ', 'VERB', 'ADV', 'PRON', 'NUM', 'DET', 'PART', 'PROPN', 'ADP', 'CCONJ', 'SCONJ'):
            if pos != 'X':
                pos = 'X'
                
        if old_pos != pos:
            changed += 1
        corrected[w] = pos
            
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(corrected, f, ensure_ascii=False, indent=2)
        
    print(f"English Maintenance Complete! Corrected/Updated {changed} tags.")

if __name__ == "__main__":
    run_maintenance()