#!/usr/bin/env python3
"""Quick verification: test triple-frame POS tagging on the problematic words
identified in the txt.txt audit.

Expected results:
- Nouns (brot, idee, volk, boot, fans, blut) → NOUN → Capitalized
- Verbs (gehe, habt, zog, höre, reitet) → VERB → lowercase
- Adjectives (müde, warm, tief, kräftig) → ADJ → lowercase
- Adverbs (voraussichtlich) → ADV → lowercase
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import spacy
except ImportError:
    print("spacy not installed, cannot run test")
    sys.exit(1)

# Load model (same preference order as the dictionary)
nlp = None
for model_name in ['de_core_news_lg', 'de_core_news_md', 'de_core_news_sm']:
    try:
        nlp = spacy.load(model_name)
        print(f"Loaded: {model_name}")
        break
    except Exception:
        continue

if nlp is None:
    print("No German SpaCy model found")
    sys.exit(1)

# Test words grouped by expected POS
test_cases = {
    'NOUN (should capitalize)': [
        'brot', 'idee', 'volk', 'boot', 'fans', 'blut', 'gast', 'tür',
        'kind', 'luft', 'mord', 'tipp', 'erde', 'liga',
    ],
    'VERB (should lowercase)': [
        'gehe', 'habt', 'zog', 'höre', 'reitet', 'eignet', 'gekrönt',
        'gekämpft', 'errichtet', 'gespeichert', 'gebrochen',
    ],
    'ADJ (should lowercase)': [
        'müde', 'warm', 'tief', 'kräftig', 'glücklich', 'lästig',
        'kritisch', 'freundlich', 'mächtig', 'heftig',
    ],
    'ADV (should lowercase)': [
        'voraussichtlich', 'bekanntlich', 'kurzfristig', 'ordentlich',
    ],
}

def get_pos_triple_frame(word_lower):
    """Triple-frame majority vote (matches new batch_tag_words logic)."""
    target_idx = 1
    
    def _get_pos(doc, w_lower):
        if len(doc) > target_idx and doc[target_idx].text.lower() == w_lower:
            return doc[target_idx].pos_
        for t in doc:
            if t.text.lower() == w_lower:
                return t.pos_
        return 'X'
    
    noun_doc = nlp(f"Das {word_lower.capitalize()} ist hier .")
    verb_doc = nlp(f"Ich {word_lower} gerne .")
    adj_doc  = nlp(f"Die {word_lower} Sachen sind gut .")
    
    noun_pos = _get_pos(noun_doc, word_lower)
    verb_pos = _get_pos(verb_doc, word_lower)
    adj_pos  = _get_pos(adj_doc, word_lower)
    
    noun_votes = sum(1 for p in [noun_pos, verb_pos, adj_pos] if p in ('NOUN', 'PROPN'))
    
    if noun_votes >= 2:
        final_pos = 'NOUN'
    else:
        if verb_pos not in ('NOUN', 'PROPN', 'X'):
            final_pos = verb_pos
        elif adj_pos not in ('NOUN', 'PROPN', 'X'):
            final_pos = adj_pos
        else:
            final_pos = noun_pos
    
    return final_pos, noun_pos, verb_pos, adj_pos

def get_pos_old_single_frame(word_lower):
    """Old single-frame approach (for comparison)."""
    doc = nlp(f"Das {word_lower.capitalize()} ist hier .")
    if len(doc) > 1 and doc[1].text.lower() == word_lower:
        return doc[1].pos_
    for t in doc:
        if t.text.lower() == word_lower:
            return t.pos_
    return 'X'

print("\n" + "=" * 100)
print(f"{'WORD':<20} {'EXPECTED':<10} {'OLD(1-frame)':<15} {'NEW(3-frame)':<15} {'noun_f':<8} {'verb_f':<8} {'adj_f':<8} {'STATUS'}")
print("=" * 100)

total = 0
fixed = 0
broken = 0
for expected_group, words in test_cases.items():
    expected_is_noun = 'NOUN' in expected_group
    print(f"\n--- {expected_group} ---")
    for w in words:
        total += 1
        old_pos = get_pos_old_single_frame(w)
        new_pos, n_pos, v_pos, a_pos = get_pos_triple_frame(w)
        
        old_correct = (old_pos in ('NOUN', 'PROPN')) == expected_is_noun
        new_correct = (new_pos in ('NOUN', 'PROPN')) == expected_is_noun
        
        if not old_correct and new_correct:
            status = "✅ FIXED"
            fixed += 1
        elif old_correct and not new_correct:
            status = "❌ BROKEN"
            broken += 1
        elif old_correct and new_correct:
            status = "✓ OK"
        else:
            status = "⚠️ STILL WRONG"
        
        print(f"  {w:<20} {'NOUN' if expected_is_noun else 'other':<10} {old_pos:<15} {new_pos:<15} {n_pos:<8} {v_pos:<8} {a_pos:<8} {status}")

print(f"\n{'=' * 100}")
print(f"Total: {total} | Fixed: {fixed} | Broken: {broken} | Net improvement: {fixed - broken}")
