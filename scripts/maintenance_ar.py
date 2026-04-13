import json
import os
import sys

# Ensure this is run from root or scripts dir
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.append(WORKSPACE_ROOT)

CACHE_FILE = os.path.join(WORKSPACE_ROOT, "models", "arabic_code", "arabic_pos_cache.json")

def run_maintenance():
    try:
        from farasa.pos import FarasaPOSTagger
    except ImportError:
        print("Please install farasapy: pip install farasapy")
        return
        
    print("Loading Farasa POSTagger...")
    tagger = FarasaPOSTagger(interactive=True)
    
    print(f"Loading Arabic cache from {CACHE_FILE}")
    if not os.path.exists(CACHE_FILE):
        print("Cache file not found!")
        return
        
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
        
    print(f"Processing {len(cache)} entries. This will enforce strict morphological tags...")
    corrected = {}
    changed = 0
    
    for w, old_pos in cache.items():
        try:
            tagged = tagger.tag(w)
            parts = tagged.split('+')
            pos = 'X'
            for part in reversed(parts):
                if '/' in part:
                    curr = part.split('/')[-1].strip().upper()
                    if curr not in ('CONJ', 'PREP', 'DET', 'PART', 'PUNC'):
                        pos = curr
                        break
            if pos == 'X' and parts and '/' in parts[-1]:
                pos = parts[-1].split('/')[-1].strip().upper()
                
            if pos == 'V': pos = 'VERB'
            elif pos == 'PREP': pos = 'ADP'
            elif pos == 'CONJ': pos = 'CCONJ'
            elif 'PRON' in pos: pos = 'PRON'
            elif pos not in ('NOUN', 'ADJ', 'ADV', 'NUM', 'DET', 'PART', 'PROPN'):
                pos = 'X'
                
            if old_pos != pos:
                changed += 1
            corrected[w] = pos
        except Exception:
            corrected[w] = 'X'
            if old_pos != 'X':
                changed += 1
            
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(corrected, f, ensure_ascii=False, indent=2)
        
    print(f"Arabic Maintenance Complete! Corrected/Updated {changed} tags.")

if __name__ == "__main__":
    run_maintenance()