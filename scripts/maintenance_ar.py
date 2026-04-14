import json
import os
import sys
import argparse
import re
from collections import Counter

# Ensure this is run from root or scripts dir
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.append(WORKSPACE_ROOT)

CACHE_FILE = os.path.join(WORKSPACE_ROOT, "models", "arabic_code", "arabic_pos_cache.json")

# Core valid tags for the Maze task
VALID_TAGS = {
    'NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'CCONJ', 
    'PRON', 'NUM', 'DET', 'PART', 'PROPN', 'SCONJ'
}

def extract_pos(tagged_word):
    """Parses Farasa morphological string and extracts the core category."""
    try:
        # 1. Clean noise markers S/S and E/E
        cleaned = tagged_word.replace('S/S ', '').replace(' E/E', '').strip()
        
        # 2. Split by / to separate morphs from tags
        # Example: "ال+ ناس/DET+NOUN-MS" -> tags = "DET+NOUN-MS"
        if '/' not in cleaned:
            return 'X'
            
        tag_sequence = cleaned.split('/')[-1].strip()
        
        # 3. Handle multiple tags joined by +
        # Example: "DET+NOUN-MS" -> ['DET', 'NOUN-MS']
        tags = tag_sequence.split('+')
        pos = 'X'
        
        # Map Farasa labels to our standards
        mapping = {
            'V': 'VERB',
            'PREP': 'ADP',
            'CONJ': 'CCONJ',
            'CASE': 'PART',  # Grammatical case markers
            'NSUFF': 'PART', 
            'VSUFF': 'PART'
        }

        # 4. Iterate through tags to find the core category
        flipped_tags = list(reversed(tags))
        for raw_tag in flipped_tags:
            # Strip sub-tags like -MS, -FP, -MP
            # Example: "NOUN-MS" -> "NOUN"
            base_tag = raw_tag.split('-')[0].upper().strip()
            
            # Map if necessary
            base_tag = mapping.get(base_tag, base_tag)
            
            # Handle PRON groups
            if 'PRON' in base_tag:
                base_tag = 'PRON'
                
            # If it's a valid "Main" tag, we take it and stop
            if base_tag in VALID_TAGS and base_tag not in ('DET', 'PART', 'PUNC', 'CCONJ'):
                pos = base_tag
                break
            
            # Keep track of the first valid minor tag as fallback
            if pos == 'X' and base_tag in VALID_TAGS:
                pos = base_tag
                
        return pos
    except Exception:
        return 'X'

def run_maintenance(dry_run=False, force=False):
    try:
        from farasa.pos import FarasaPOSTagger
    except ImportError:
        print("Please install farasapy: pip install farasapy")
        return
        
    print(f"--- Arabic Cache Maintenance {'(DRY RUN)' if dry_run else ''} ---")
    
    if not os.path.exists(CACHE_FILE):
        print(f"Error: {CACHE_FILE} not found!")
        return
        
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
        
    print(f"Processing {len(cache)} entries using Farasa POSTagger...")
    tagger = FarasaPOSTagger(interactive=True)
    
    corrected = {}
    stats = Counter()
    old_stats = Counter(cache.values())
    changed = 0
    
    # Debug counter
    debug_shown = 0
    
    for word, old_pos in cache.items():
        try:
            tagged = tagger.tag(word)
            new_pos = extract_pos(tagged)
            
            if new_pos == "X" and stats["X"] < 10:
                print(f"[X-DIAG] word='{word}' raw='{tagged}'")

            if debug_shown < 10:
                print(f"DEBUG: word='{word}' raw='{tagged}' -> extracted='{new_pos}'")
                debug_shown += 1
            
            if old_pos != new_pos:
                changed += 1
            
            corrected[word] = new_pos
            stats[new_pos] += 1
        except Exception as e:
            print(f"CRITICAL ERROR tagging '{word}': {e}")
            corrected[word] = 'X'
            stats['X'] += 1
            if old_pos != 'X':
                changed += 1

    # --- SUMMARY ---
    total = len(cache)
    print("\nLinguistic Summary:")
    for tag in sorted(stats.keys()):
        print(f"  {tag:6}: {stats[tag]:5} (was {old_stats.get(tag, 0)})")
        
    print(f"\nTotal Updates: {changed}")
    x_percentage = (stats['X'] / total) * 100
    print(f"Failure Rate ('X'): {x_percentage:.2f}% ({stats['X']} words)")

    if dry_run:
        print("\n--- Dry Run Complete. No changes saved. ---")
    else:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(corrected, f, ensure_ascii=False, indent=2)
        print("\n--- Cache File Successfully Updated! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and Maintain Arabic POS Cache")
    parser.add_argument("--dry-run", action="store_true", help="Audit tags without saving to disk")
    parser.add_argument("--force", action="store_true", help="Ignore Safety Gate and save anyway")
    args = parser.parse_args()
    
    run_maintenance(dry_run=args.dry_run, force=args.force)