import json
import os
import sys

# Ensure this is run from root or scripts dir
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.append(WORKSPACE_ROOT)

CACHE_FILE = os.path.join(WORKSPACE_ROOT, "models", "german_code", "german_pos_cache_v2.json")

def run_maintenance():
    try:
        from HanTa import HanoverTagger as ht
    except ImportError:
        print("Please install HanTa: pip install HanTa")
        return
        
    print("Loading HanTa Morphological Dictionary...")
    tagger = ht.HanoverTagger('morphmodel_ger.pgz')
    
    print(f"Loading German cache from {CACHE_FILE}")
    if not os.path.exists(CACHE_FILE):
        print("Cache file not found!")
        return
        
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
        
    stts_map = {
        'NN': 'NOUN', 'NE': 'PROPN',
        'ADJA': 'ADJ', 'ADJD': 'ADJ',
        'VVFIN': 'VERB', 'VVIMP': 'VERB', 'VVINF': 'VERB', 'VVIZU': 'VERB', 'VVPP': 'VERB',
        'VMFIN': 'VERB', 'VMINF': 'VERB', 'VMPP': 'VERB',
        'VAFIN': 'AUX', 'VAIMP': 'AUX', 'VAINF': 'AUX', 'VAPP': 'AUX',
        'ADV': 'ADV', 'PROAV': 'ADV', 'PTKA': 'ADV',
        'APPR': 'ADP', 'APPRART': 'ADP', 'APPO': 'ADP', 'APZR': 'ADP',
        'ART': 'DET', 'PIAT': 'DET', 'PDAT': 'DET', 'PPOSAT': 'DET', 'PWAT': 'DET',
        'PPER': 'PRON', 'PRF': 'PRON', 'PDS': 'PRON', 'PIS': 'PRON', 'PPOSS': 'PRON', 'PRELS': 'PRON', 'PWS': 'PRON',
        'CARD': 'NUM', 'KON': 'CCONJ', 'KOUS': 'SCONJ', 'KOUI': 'SCONJ',
        'PTKNEG': 'PART', 'PTKVZ': 'PART', 'PTKANT': 'PART', 'PTKZU': 'PART',
        'ITJ': 'INTJ'
    }
    
    print(f"Processing {len(cache)} entries. This will enforce strict HanTa tagging...")
    corrected = {}
    changed = 0
    
    for w, old_pos in cache.items():
        try:
            # Enforce capitalizing to test morphological compatibility
            w_cap = w.capitalize()
            tags_cap = tagger.tag_word(w_cap)
            tags_low = tagger.tag_word(w.lower())
            
            clean_tag_cap = tags_cap[0][0].replace('(', '').replace(')', '') if tags_cap else 'X'
            clean_tag_low = tags_low[0][0].replace('(', '').replace(')', '') if tags_low else 'X'
            
            score_cap = float(tags_cap[0][1]) if tags_cap else -100.0
            score_low = float(tags_low[0][1]) if tags_low else -100.0
            
            pos = stts_map.get(clean_tag_cap, 'X')
            
            if pos == 'NOUN':
                upos_low = stts_map.get(clean_tag_low, 'X')
                if upos_low in ('VERB', 'ADJ', 'ADV', 'AUX') and score_low > -20.0:
                    pos = upos_low
            else:
                # Use the lowercase tag directly for non-nouns
                pos = stts_map.get(clean_tag_low, 'X')
                
            if pos not in ('NOUN', 'ADJ', 'VERB', 'ADV', 'PRON', 'NUM', 'DET', 'PART', 'PROPN', 'ADP', 'CCONJ', 'SCONJ', 'AUX', 'INTJ'):
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
        
    print(f"German Maintenance Complete! Corrected/Updated {changed} tags.")

if __name__ == "__main__":
    run_maintenance()