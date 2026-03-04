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
            matches.append(word.text)
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
        logging.warning(
            "Could not find enough distractors for requested num_to_test; "
            "continuing with smaller candidate pool (non-fatal)"
        )
        return distractor_opts



class wordfreq_English_zipf_dict(wordfreq_dict):
    """Zipf-based English dictionary with German-style filtering knobs.

    Supported params:
    - min_zipf (float, default 3.0)
    - min_word_len (int, default 3)
    - lowercase_only (bool, default True)
    - include_words (path, optional)
    - exclude_words (path, optional)
    """

    def __init__(self, params={}):
        self.lang = "en"
        exclude = params.get("exclude_words", "exclude_en.txt")
        include = params.get("include_words", None)
        lowercase_only = bool(params.get("lowercase_only", True))
        min_word_len = int(params.get("min_word_len", 3))
        min_zipf = float(params.get("min_zipf", 3.0))

        exclusions_lower = set()
        if exclude is not None:
            try:
                with open(exclude, "r", encoding="utf-8") as f:
                    exclusions_lower = set(line.strip().lower() for line in f if line.strip())
            except Exception:
                exclusions_lower = set()

        include_words = None
        if include is not None and os.path.exists(include):
            try:
                with open(include, "r", encoding="utf-8") as f:
                    include_words = [line.strip() for line in f if line.strip()]
            except Exception:
                include_words = None

        freq_dict = wordfreq.get_frequency_dict("en")
        source_words = include_words if include_words is not None else freq_dict.keys()

        self.words = []
        seen = set()
        alpha_re = r"^[a-z]+$" if lowercase_only else r"^[A-Za-z]+$"
        vowel_re = r"[aeiouyAEIOUY]"
        for raw in source_words:
            token = raw.lower() if lowercase_only else raw
            low = token.lower()
            if low in seen:
                continue
            if low in exclusions_lower:
                continue
            if not re.match(alpha_re, token):
                continue
            if len(token) < min_word_len:
                continue
            if not re.search(vowel_re, token):
                continue
            try:
                z = wordfreq.zipf_frequency(token, "en")
            except Exception:
                continue
            if z < min_zipf:
                continue
            freq_val = z * math.log(10)
            self.words.append(distractor(token, freq_val))
            seen.add(low)


# ---------------------------------------------------------------------------
# German noun suffix heuristic (no spaCy required)
# ---------------------------------------------------------------------------
# Well-known derivational suffixes that almost exclusively form nouns.
_GERMAN_NOUN_SUFFIXES = (
    'ung', 'heit', 'keit', 'schaft', 'tion', 'sion', 'nis', 'tum',
    'ling', 'tät', 'ment', 'chen', 'lein', 'ismus', 'eur', 'ant',
    'ent', 'ist', 'enz', 'anz',
)

def _is_likely_german_noun(word):
    """Heuristic: return True if *word* looks like a German noun.

    Uses a curated list of derivational suffixes that almost exclusively
    produce nouns in German.  This is a reliable fallback when no spaCy
    model is installed.
    """
    lw = word.lower()
    if len(lw) < 3:
        return False
    for suffix in _GERMAN_NOUN_SUFFIXES:
        if lw.endswith(suffix) and len(lw) > len(suffix):
            return True
    return False


class wordfreq_German_zipf_dict(wordfreq_dict):
    """Zipf-based German dictionary built from the wordfreq library.

    Frequencies stored on distractor objects are in natural-log units
    (zipf * ln(10)) to be consistent with other code.
    German noun casing is handled by apply_postcase in sentence_set.py
    (spaCy POS tagging + .capitalize()).

    Supported params:
    - min_zipf (float, default 3.0)
    - min_word_len (int, default 3)
    - lowercase_only (bool, default True)
    - include_words (path, optional)
    - exclude_words (path, optional)
    """

    def __init__(self, params={}):
        self.lang = "de"
        exclude = params.get("exclude_words", "exclude_de.txt")
        include = params.get("include_words", None)
        lowercase_only = bool(params.get("lowercase_only", True))
        min_word_len = int(params.get("min_word_len", 3))
        min_zipf = float(params.get("min_zipf", 3.0))

        exclusions_lower = set()
        if exclude is not None:
            try:
                with open(exclude, "r", encoding="utf-8") as f:
                    for line in f:
                        w = line.strip()
                        if w and not w.startswith("#"):
                            exclusions_lower.add(w.lower())
            except Exception:
                exclusions_lower = set()

        include_words = None
        if include is not None and os.path.exists(include):
            try:
                with open(include, "r", encoding="utf-8") as f:
                    include_words = [line.strip() for line in f if line.strip()]
            except Exception:
                include_words = None

        freq_dict = wordfreq.get_frequency_dict("de")
        source_words = include_words if include_words is not None else freq_dict.keys()

        self.words = []
        seen = set()
        for raw in source_words:
            w = raw.strip()
            lw = w.lower()
            if lw in seen:
                continue
            if lw in exclusions_lower:
                continue
            if not re.match(r"^[a-zäöüß]+$", lw):
                continue
            if len(lw) < min_word_len:
                continue
            # Require a vowel so candidates are clearly word-like
            if not re.search(r"[aeiouyäöü]", lw):
                continue
            # Reject likely abbreviations: short words with very few vowels
            # (e.g., nsu, olg, edv, sap, vip, akp). Real German words of
            # length 3-4 almost always have >= 2 vowels or >= 50% vowel ratio.
            vowel_count = len(re.findall(r"[aeiouyäöü]", lw))
            if len(lw) <= 4 and vowel_count <= 1:
                continue
            if _is_english_dominant(lw):
                continue
            try:
                z = wordfreq.zipf_frequency(lw, "de")
            except Exception:
                continue
            if z < min_zipf:
                continue
            freq_val = z * math.log(10)
            self.words.append(distractor(lw, freq_val))
            seen.add(lw)

        # Build case_map: identify which distractor words are German nouns
        # and store their Titlecased form.
        self.case_map = {}
        self._build_case_map()

    def _build_case_map(self):
        """Identify nouns via 3-context majority-vote spaCy tagging.

        Each word is tagged in three sentence contexts:
          1. Isolated (just the word)
          2. Neutral:    'Das {word} ist hier.'
          3. Verb-biased: 'Er will {word} gehen.'

        If >= 2 of the 3 contexts tag the word as NOUN/PROPN, it's a noun.
        Falls back to suffix heuristic when no spaCy model is available.
        """
        all_words = [d_obj.text for d_obj in self.words]
        if not all_words:
            return

        nlp_sp = None
        try:
            import spacy
            try:
                nlp_sp = spacy.load('de_core_news_sm')
            except Exception:
                try:
                    nlp_sp = spacy.load('de_core_news_md')
                except Exception:
                    nlp_sp = None
        except Exception:
            nlp_sp = None

        if nlp_sp is not None:
            # Batch-tag in three contexts for majority vote
            iso_texts = all_words
            neutral_texts = [f'Das {w} ist hier.' for w in all_words]
            verb_texts = [f'Er will {w} gehen.' for w in all_words]

            def _batch_noun_flags(texts, word_idx):
                """Return list of bools: True if token at word_idx is NOUN/PROPN."""
                flags = []
                for doc in nlp_sp.pipe(texts, batch_size=256):
                    tokens = list(doc)
                    if word_idx < len(tokens):
                        flags.append(tokens[word_idx].pos_ in ('NOUN', 'PROPN'))
                    else:
                        flags.append(False)
                return flags

            iso_flags = _batch_noun_flags(iso_texts, 0)
            neutral_flags = _batch_noun_flags(neutral_texts, 1)  # 'Das {word} ...'
            verb_flags = _batch_noun_flags(verb_texts, 2)        # 'Er will {word} ...'

            for i, w in enumerate(all_words):
                votes = sum([iso_flags[i], neutral_flags[i], verb_flags[i]])
                if votes >= 2:
                    self.case_map[w.lower()] = w.lower().capitalize()
        else:
            # Fallback: suffix heuristic (less accurate but works without spaCy)
            for w in all_words:
                if _is_likely_german_noun(w):
                    self.case_map[w.lower()] = w.lower().capitalize()

    def canonical_case(self, token):
        """Return Titlecased form if known noun, else token as-is."""
        return self.case_map.get(token.lower(), token)

    def get_titlecase_variant(self, token):
        """Return Titlecased variant if the word is a known German noun."""
        return self.case_map.get(token.lower(), None)


def get_frequency_de(word):
    """Returns German Zipf frequency converted to natural-log units."""
    return wordfreq.zipf_frequency(word, 'de') * math.log(10)


def get_frequency_en(word):
    """Returns English Zipf frequency converted to natural-log units."""
    return wordfreq.zipf_frequency(word, 'en') * math.log(10)




def get_thresholds_de(words):
    """German thresholds based on German frequency."""
    lengths = []
    freqs = []
    for word in words:
        stripped = utils.strip_punct(word)
        # Clamp word lengths to Boyce-style bins [3, 15] before range creation.
        lengths.append(max(3, min(len(stripped), 15)))
        freqs.append(get_frequency_de(stripped))
    min_length = min(lengths)
    max_length = max(lengths)
    min_freq = min(min(freqs), 11)
    max_freq = max(max(freqs), 3)
    return min_length, max_length, min_freq, max_freq


def get_thresholds_en(words):
    """English thresholds based on English frequency."""
    lengths = []
    freqs = []
    for word in words:
        stripped = utils.strip_punct(word)
        # Clamp word lengths to Boyce-style bins [3, 15] before range creation.
        lengths.append(max(3, min(len(stripped), 15)))
        freqs.append(get_frequency_en(stripped))
    min_length = min(lengths)
    max_length = max(lengths)
    min_freq = min(min(freqs), 11)
    max_freq = max(max(freqs), 3)
    return min_length, max_length, min_freq, max_freq


# ---------------------------------------------------------------------------
# Arabic support
# ---------------------------------------------------------------------------

# Arabic diacritical marks (tashkeel) Unicode range: U+0610–U+061A, U+064B–U+065F, U+0670
_ARABIC_DIACRITICS_RE = re.compile(
    r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]'
)


def strip_arabic_diacritics(text):
    """Remove Arabic diacritical marks (tashkeel/harakat) from text."""
    return _ARABIC_DIACRITICS_RE.sub('', text)


# Regex for Arabic-only word tokens (core Arabic block).
_ARABIC_WORD_RE = re.compile(r'^[\u0600-\u06FF]+$')


class wordfreq_Arabic_zipf_dict(wordfreq_dict):
    """Zipf-based Arabic dictionary built from the wordfreq library.

    Arabic has no uppercase/lowercase distinction, so casing logic is skipped.
    Diacritics (tashkeel) are stripped for consistent matching.
    """

    def __init__(self, params={}):
        self.lang = "ar"
        exclude = params.get("exclude_words", "exclude_ar.txt")
        include = params.get("include_words", None)
        min_word_len = int(params.get("min_word_len", 2))
        min_zipf = float(params.get("min_zipf", 3.0))

        exclusions_lower = set()
        if exclude is not None:
            try:
                with open(exclude, "r", encoding="utf-8") as f:
                    exclusions_lower = set(
                        strip_arabic_diacritics(line.strip()) for line in f if line.strip()
                    )
            except Exception:
                exclusions_lower = set()

        include_words = None
        if include is not None and os.path.exists(include):
            try:
                with open(include, "r", encoding="utf-8") as f:
                    include_words = [line.strip() for line in f if line.strip()]
            except Exception:
                include_words = None

        freq_dict = wordfreq.get_frequency_dict("ar")
        source_words = include_words if include_words is not None else freq_dict.keys()

        self.words = []
        seen = set()
        for raw in source_words:
            token = strip_arabic_diacritics(raw)
            if token in seen:
                continue
            if token in exclusions_lower:
                continue
            if not _ARABIC_WORD_RE.match(token):
                continue
            if len(token) < min_word_len:
                continue
            try:
                z = wordfreq.zipf_frequency(token, "ar")
            except Exception:
                continue
            if z < min_zipf:
                continue
            freq_val = z * math.log(10)
            self.words.append(distractor(token, freq_val))
            seen.add(token)

    def canonical_case(self, token):
        """Arabic has no casing; return as-is."""
        return token

    def get_titlecase_variant(self, token):
        """Arabic has no casing; always returns None."""
        return None


def get_frequency_ar(word):
    """Returns Arabic Zipf frequency converted to natural-log units."""
    return wordfreq.zipf_frequency(strip_arabic_diacritics(word), 'ar') * math.log(10)


def get_thresholds_ar(words):
    """Arabic thresholds based on Arabic frequency."""
    lengths = []
    freqs = []
    for word in words:
        stripped = utils.strip_punct(word)
        # Arabic words can be shorter; clamp to [2, 15].
        lengths.append(max(2, min(len(stripped), 15)))
        freqs.append(get_frequency_ar(stripped))
    min_length = min(lengths)
    max_length = max(lengths)
    min_freq = min(min(freqs), 11)
    max_freq = max(max(freqs), 3)
    return min_length, max_length, min_freq, max_freq

