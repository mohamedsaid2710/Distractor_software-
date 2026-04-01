#!/usr/bin/env python3
"""Test script to verify POS tagging accuracy improvements.

Run this on Colab after installing requirements to verify:
1. Large SpaCy model loaded correctly
2. Contextual framing improves adjective detection
3. POS filtering works as expected
"""

import sys
sys.path.insert(0, '.')
from wordfreq_distractor import wordfreq_German_zipf_dict

# Problematic adjectives that were misclassified before
TEST_ADJECTIVES = [
    'fränkischen', 'württembergischen', 'europäischen', 'politischen',
    'münchner', 'österreichischen', 'historischen', 'deutschen'
]

# Clear nouns for comparison
TEST_NOUNS = [
    'anwältin', 'musikerin', 'lektorin', 'volksverhetzung',
    'parlament', 'regierung', 'universität'
]

# Initialize dictionary
params = {
    'exclude_words': 'exclude_de.txt',
    'min_zipf': 3.5,
    'min_word_len': 2,
    'num_to_test': 100
}

print("="*70)
print("POS TAGGING ACCURACY TEST")
print("="*70)
print()

d = wordfreq_German_zipf_dict(params)

# Check which model was loaded
if d.nlp_sp:
    model_name = d.nlp_sp.meta.get('name', 'unknown')
    print(f"✓ SpaCy model loaded: {model_name}")
else:
    print("✗ No SpaCy model loaded!")
    sys.exit(1)

print()

# Test adjectives
print("Testing Adjectives (should be tagged as ADJ):")
print("-" * 70)
d.batch_tag_words(TEST_ADJECTIVES)
adj_correct = 0
for w in TEST_ADJECTIVES:
    pos = d.pos_cache.get(w.lower(), 'UNKNOWN')
    is_correct = pos == 'ADJ'
    symbol = "✓" if is_correct else "✗"
    adj_correct += is_correct
    print(f"{symbol} {w:25s} → POS={pos:8s}")

adj_accuracy = (adj_correct / len(TEST_ADJECTIVES)) * 100
print(f"\nAdjective Accuracy: {adj_correct}/{len(TEST_ADJECTIVES)} ({adj_accuracy:.1f}%)")

print()

# Test nouns
print("Testing Nouns (should be tagged as NOUN/PROPN):")
print("-" * 70)
d.batch_tag_words(TEST_NOUNS)
noun_correct = 0
for w in TEST_NOUNS:
    pos = d.pos_cache.get(w.lower(), 'UNKNOWN')
    is_correct = pos in ('NOUN', 'PROPN')
    symbol = "✓" if is_correct else "✗"
    noun_correct += is_correct
    print(f"{symbol} {w:25s} → POS={pos:8s}")

noun_accuracy = (noun_correct / len(TEST_NOUNS)) * 100
print(f"\nNoun Accuracy: {noun_correct}/{len(TEST_NOUNS)} ({noun_accuracy:.1f}%)")

print()
print("="*70)
overall_correct = adj_correct + noun_correct
overall_total = len(TEST_ADJECTIVES) + len(TEST_NOUNS)
overall_accuracy = (overall_correct / overall_total) * 100
print(f"OVERALL ACCURACY: {overall_correct}/{overall_total} ({overall_accuracy:.1f}%)")
print("="*70)

if overall_accuracy >= 95:
    print("✓ EXCELLENT - POS filtering will work very accurately")
elif overall_accuracy >= 85:
    print("✓ GOOD - POS filtering will work well with minor errors")
elif overall_accuracy >= 70:
    print("⚠ FAIR - Some adjectives may slip through, consider suffix rules")
else:
    print("✗ POOR - Consider reverting to small model + suffix rules")
