import wordfreq
import os
import re
import math
import random
import logging
import json

import utils
from distractor import distractor_dict, distractor


"""
Filter out non-German tokens that can appear in German vocab sources.
The helper below drops words that are much more frequent in English than German.
"""
def _is_english_dominant(token: str, margin: float = 0.3) -> bool:
    """
    Return True if the word's English Zipf frequency exceeds German by `margin`. Uses wordfreq; any lookup failures return False.
    """
    try:
        en = wordfreq.zipf_frequency(token, 'en')
        de = wordfreq.zipf_frequency(token, 'de')
        return (en - de) > margin
    except Exception:
        return False


class wordfreq_dict(distractor_dict):
    """General class of dictionaries"""

    def __init__(self, params={}):
        pass
    def canonical_case(self, token):
        """Return a preferred-cased form of `token` if available (override in subclasses)."""
        return token
    def in_dict(self, test_word):
        """Test to see if word is in dictionary"""
        for word in self.words:
            if word.text == test_word:
                return word
        return False

    def get_words(self, length_low, length_high, freq_low, freq_high):
        """Returns a list of words within specified ranges"""
        matches = []
        for word in self.words:
            # basic range checks
            if not (freq_low <= word.freq <= freq_high and length_low <= word.len <= length_high):
                continue
            w = word.text
            # filter: drop short all-caps acronyms (e.g., 'SPD', 'EU')
            try:
                if w.isupper() and len(w) <= 4:
                    continue
            except Exception:
                pass
            # filter: drop tokens that are more frequent in English than German
            try:
                en_z = wordfreq.zipf_frequency(w, 'en')
                de_z = wordfreq.zipf_frequency(w, 'de')
                # if English frequency notably higher than German, skip
                if en_z - de_z > 0.3:
                    continue
            except Exception:
                pass
            matches.append(w)
        return matches

    def get_potential_distractors(self, min_length, max_length, min_freq, max_freq, params):
        """returns list of n words, if possible from between threshold values
        if not tries things nearby -- higher frequency and then lower"""
        distractor_opts = self.get_words(min_length, max_length, min_freq, max_freq)
        random.shuffle(distractor_opts)
        n=params['num_to_test']
        if len(distractor_opts) >= n:
            return distractor_opts[:n]
        else:
            logging.info("Having to widen distractor option search")
            still_need = n - len(distractor_opts)
            i = 1
            while i < 10:
                new = []
                lower = self.get_words(min_length, max_length, min_freq - i, min_freq - i + 1)
                higher = self.get_words(min_length, max_length, max_freq + i - 1, max_freq + i)
                new.extend(lower)
                new.extend(higher)
                random.shuffle(new)
                if len(new) >= still_need:
                    distractor_opts.extend(new)
                    return distractor_opts[:n]
                distractor_opts.extend(new)
                i += 1
        logging.warning("Could not find enough distractors")
        return distractor_opts


class wordfreq_English_dict(wordfreq_dict):
    """Dictionary built using word freq for frequencies
     Words need to be in wordfreq's vocab, also in include file if provided
     and not in exclude file
     words must be lowercase alpha only"""

    def __init__(self, params={}):
        exclude = params.get("exclude_words", "exclude.txt")
        include = params.get("include_words", None)
        dict = wordfreq.get_frequency_dict('en')
        keys = dict.keys()
        self.words = []
        exclusions = []

        if exclude is not None:
            with open(exclude, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    exclusions.append(word)
        inclusions = []
        if include is not None and os.path.exists(include):
            with open(include, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    inclusions.append(word)
            words = list(set(inclusions) & set(keys) - set(exclusions))
        else:
            words = list(set(keys) - set(exclusions))
        for word in words:
            if re.match("^[a-z]*$", word):
                freq = math.log(
                    dict[word] * 10 ** 9)  # we canonically calculate frequency as log occurrences/1 billion words
                self.words.append(distractor(word, freq))


class wordfreq_German_zipf_dict(wordfreq_dict):
    """Zipf-based German dictionary loaded from wordfreq_de.tsv.
    Frequencies stored on distractor objects are in natural-log units (zipf * ln(10))
    to be consistent with other code.
    """

    def __init__(self, params={}):
        exclude = params.get("exclude_words", "exclude.txt")
        include = params.get("include_words", None)
        tsv_path = os.path.join(os.path.dirname(__file__), 'data', 'german_data', 'wordfreq_de.tsv')
        data_map = None
        
        # Load from wordfreq_de.tsv (TSV format: word\tzipf)
        if os.path.exists(tsv_path):
            try:
                data_map = {}
                with open(tsv_path, 'r', encoding='utf-8') as f:
                    next(f)  # skip header
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            orig = parts[0].strip()
                            tok = orig.lower()
                            try:
                                z = float(parts[1])
                            except Exception:
                                continue
                            data_map[tok] = (orig, z)
            except Exception:
                data_map = None

        # Respect `min_zipf` parameter to avoid selecting very rare tokens
        MIN_ZIPF = params.get('min_zipf', 3.0)

        # build word list from data_map or fallback to wordfreq
        self.words = []
        exclusions = []
        if exclude is not None:
            try:
                with open(exclude, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip()
                        if word:
                            exclusions.append(word)
            except Exception:
                exclusions = []
        exclusions_lower = set([e.lower() for e in exclusions])

        if data_map is not None:
            for lower, (orig, z) in data_map.items():
                if lower in exclusions_lower:
                    continue
                if not re.match(r"^[A-Za-zÄÖÜäöüß]+$", orig):
                    continue
                if len(orig) < 3:
                    continue
                if _is_english_dominant(orig):
                    continue
                # enforce min zipf threshold
                try:
                    if float(z) < MIN_ZIPF:
                        continue
                except Exception:
                    continue
                # store freq as ln(count) equivalent: zipf*ln(10)
                try:
                    freq_val = float(z) * math.log(10)
                except Exception:
                    continue
                self.words.append(distractor(orig, freq_val))
        else:
            # final fallback: use wordfreq zipf and convert
            dict_map = wordfreq.get_frequency_dict('de')
            for w in dict_map.keys():
                lw = w.lower()
                if lw in exclusions_lower:
                    continue
                if not re.match(r"^[a-zäöüß]+$", w.lower()):
                    continue
                try:
                    z = wordfreq.zipf_frequency(w, 'de')
                    # enforce min zipf threshold for fallback source as well
                    if z < MIN_ZIPF:
                        continue
                    if _is_english_dominant(w):
                        continue
                    freq_val = z * math.log(10)
                    self.words.append(distractor(w, freq_val))
                except Exception:
                    continue

        # build case_map for titlecase lookup
        try:
            case_map = {}
            if data_map is not None:
                for k, (orig, _) in data_map.items():
                    case_map[k] = orig
            self.case_map = case_map
        except Exception:
            self.case_map = {}

    def canonical_case(self, token):
        low = token.lower()
        return self.case_map.get(low, token)

    def get_titlecase_variant(self, token):
        low = token.lower()
        cand = self.case_map.get(low)
        if cand and len(cand) > 1 and cand[0].isupper() and cand[1:].islower():
            return cand
        return None


def get_frequency_de(word):
    """Returns German frequency from wordfreq_de.tsv or wordfreq fallback."""
    tsv_path = os.path.join(os.path.dirname(__file__), 'data', 'german_data', 'wordfreq_de.tsv')
    w = word.lower()
    # Try wordfreq_de.tsv
    if os.path.exists(tsv_path):
        try:
            with open(tsv_path, 'r', encoding='utf-8') as f:
                next(f)  # skip header
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        tok = parts[0].strip().lower()
                        if tok == w:
                            try:
                                return float(parts[1]) * math.log(10)
                            except Exception:
                                break
        except Exception:
            pass
    # fallback to wordfreq library
    return wordfreq.zipf_frequency(word, 'de') * math.log(10)


def get_frequency_en(word):
    """Returns English Zipf frequency converted to natural-log units."""
    return wordfreq.zipf_frequency(word, 'en') * math.log(10)


def get_frequency(word):
    """Backward-compatible alias: German frequency."""
    return get_frequency_de(word)


def get_thresholds(words):
    """German thresholds based on German frequency."""
    lengths = []
    freqs = []
    for word in words:
        stripped = utils.strip_punct(word)
        lengths.append(len(stripped))
        freqs.append(get_frequency_de(stripped))
    min_length = min(min(lengths), 15)
    max_length = max(max(lengths), 4)
    min_freq = min(min(freqs), 11)
    max_freq = max(max(freqs), 3)
    return min_length, max_length, min_freq, max_freq


def get_thresholds_en(words):
    """English thresholds based on English frequency."""
    lengths = []
    freqs = []
    for word in words:
        stripped = utils.strip_punct(word)
        lengths.append(len(stripped))
        freqs.append(get_frequency_en(stripped))
    min_length = min(min(lengths), 15)
    max_length = max(max(lengths), 4)
    min_freq = min(min(freqs), 11)
    max_freq = max(max(freqs), 3)
    return min_length, max_length, min_freq, max_freq
