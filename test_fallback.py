import sys
sys.path.append('.')

from wordfreq_distractor import wordfreq_German_zipf_dict

d = wordfreq_German_zipf_dict()
print("Dict loaded.")
# Let's inspect the POS tag for some of the wrong distractors
words_to_test = ["augenmerk", "häuschen", "gefährden", "nüsse", "strafe", "nahe", "herzogin", "verhandlung", "prävention", "geborenen", "erkrankte", "anleihen", "märz", "zufolge", "rädern", "blut", "verbrennung", "angestellte", "verträgen", "kapazität", "anzugreifen"]

for w in words_to_test:
    # Look it up in dictionary pos cache
    tag = d.pos_cache.get(w, "NOT_IN_CACHE")
    print(f"{w}: {tag}")

