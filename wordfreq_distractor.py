import wordfreq
import os
import re
import math
import random
import logging
import json

from utils import strip_punct
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
        self.words = []
        self.words_by_len = {}

    def _build_length_index(self):
        """Internal helper to organize words by length for fast lookup."""
        self.words_by_len = {}
        for w in self.words:
            l = w.len
            if l not in self.words_by_len:
                self.words_by_len[l] = []
            self.words_by_len[l].append(w)

    def canonical_case(self, token):
        """Return a preferred-cased form of `token` if available (override in subclasses)."""
        return token

    def in_dict(self, test_word):
        """Test to see if word is in dictionary"""
        # Minor optimization: first check length
        l = len(strip_punct(test_word))
        word_pool = self.words_by_len.get(l, [])
        for word in word_pool:
            if word.text == test_word:
                return word
        return False

    def get_words(self, length_low, length_high, freq_low, freq_high, pos_filter=None, use_spacy=False):
        """Returns a list of words within specified ranges using length-based indexing."""
        matches = []
        
        # Iterate only over the relevant length buckets
        for l in range(length_low, length_high + 1):
            word_pool = self.words_by_len.get(l, [])
            for word in word_pool:
                # Basic frequency check
                if not (freq_low <= word.freq <= freq_high):
                    continue
                
                if pos_filter:
                    p_tag = getattr(word, 'pos', None)
                    # For German, lazy-evaluation is now handled via batch processing in get_potential_distractors.
                    # This fallback is kept for robustness but should rarely be bottlenecked now.
                    if use_spacy and hasattr(self, 'has_titlecase_variant'):
                        if p_tag != pos_filter:
                            is_noun = self.has_titlecase_variant(word.text)
                            p_tag = "NOUN" if is_noun else None

                    if pos_filter.startswith('!'):
                        if p_tag == pos_filter[1:]:
                            continue
                    elif p_tag != pos_filter:
                        continue
                matches.append(word.text)
        return matches

    def batch_tag_words(self, words, batch_size=512):
        """Tag a list of words in bulk using high-performance SpaCy batching.
        
        Evaluates capitalization mapping to safely identify valid German nouns.
        
        Args:
            words: List of words to tag
            batch_size: SpaCy pipeline batch size (default 512, configurable via spacy_batch_size param)
        """
        if self.nlp_sp is None or not words:
            return
        
        # German function words - ALWAYS lowercase
        FUNCTION_WORDS = {
            'ins', 'im', 'am', 'ans', 'zum', 'zur', 'vom', 'beim', 'durchs', 'fürs',
            'ums', 'aufs', 'übers', 'unters', 'hinters', 'vors',
            'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einem', 'eines', 'einer', 'einen',
            'mein', 'dein', 'sein', 'ihr', 'unser', 'euer',
            'ab', 'an', 'auf', 'aus', 'bei', 'bis', 'durch', 'für', 'gegen',
            'hinter', 'in', 'mit', 'nach', 'neben', 'ohne', 'über', 'um',
            'unter', 'von', 'vor', 'zu', 'zwischen',
            'pro', 'per', 'ach', 'oh'
        }
        
        # Initialize POS cache if not exists
        if not hasattr(self, 'pos_cache'):
            self.pos_cache = {}
        
        # Filter for words not yet in case_map
        unique_words = list(set(w.lower() for w in words if w.lower() not in self.case_map))
        if not unique_words:
            return
        
        # First pass: mark function words
        for w in unique_words:
            if w in FUNCTION_WORDS:
                self.case_map[w] = None
                self.pos_cache[w] = 'ADP'
        
        # Second pass: SpaCy tagging with CAPITALIZED mapping 
        content_words = [w for w in unique_words if w not in FUNCTION_WORDS]
        if content_words:
            cap_words = [w.capitalize() for w in content_words]
            try:
                # High-performance batched pipeline (keep lemmatizer for morphological analysis)
                docs = list(self.nlp_sp.pipe(cap_words, disable=['ner', 'parser'], batch_size=batch_size))
                for word_l, cap_word, doc in zip(content_words, cap_words, docs):
                    if len(doc) > 0 and doc[0].pos_ in ('NOUN', 'PROPN'):
                        self.pos_cache[word_l] = doc[0].pos_
                        self.case_map[word_l] = cap_word
                    else:
                        self.pos_cache[word_l] = doc[0].pos_ if len(doc) > 0 else 'X'
                        self.case_map[word_l] = None
            except Exception as e:
                logging.error(f"SpaCy batch tagging failed: {e}")
                for word_l in content_words:
                    self.case_map[word_l] = None



    def get_potential_distractors(self, min_length, max_length, min_freq, max_freq, params, pos_filter=None):
        """Returns list of candidates, using heuristic first, then widening, then batch SpaCy validation."""
        n = params.get('num_to_test', 200)
        target_pool_size = max(n * 2, 500)
        
        # Get exclude list from params for pre-filtering
        exclude_words_set = set()
        exclude_path = params.get('exclude_words', None)
        if exclude_path:
            import os
            if not os.path.isabs(exclude_path) and not os.path.exists(exclude_path):
                base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                exclude_path = os.path.join(base, exclude_path)
            if os.path.exists(exclude_path):
                try:
                    with open(exclude_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            w = line.strip().lower()
                            if w and not w.startswith('#'):
                                exclude_words_set.add(w)
                except Exception as e:
                    logging.warning(f"Failed to load exclude list from {exclude_path}: {e}")
        
        # 1. Fetch search pool (using heuristic only)
        # We fetch MORE so that after POS filtering we still have 'n' candidates.
        distractor_opts = self.get_words(min_length, max_length, min_freq, max_freq, pos_filter=None, use_spacy=False)
        
        # PRE-FILTER: Remove excluded words BEFORE any further processing
        if exclude_words_set:
            distractor_opts = [w for w in distractor_opts if strip_punct(w).lower() not in exclude_words_set]
        
        # 2. Widening frequency range if needed (still heuristic/raw)
        if len(distractor_opts) < target_pool_size:
            max_widen = int(params.get('max_freq_widen', 15))
            for i in range(1, max_widen + 1):
                lower = self.get_words(min_length, max_length, min_freq - i, min_freq - i + 1, pos_filter=None, use_spacy=False)
                higher = self.get_words(min_length, max_length, max_freq + i - 1, max_freq + i, pos_filter=None, use_spacy=False)
                # Also filter the widened pools
                if exclude_words_set:
                    lower = [w for w in lower if strip_punct(w).lower() not in exclude_words_set]
                    higher = [w for w in higher if strip_punct(w).lower() not in exclude_words_set]
                distractor_opts.extend(lower)
                distractor_opts.extend(higher)
                if len(distractor_opts) >= target_pool_size:
                    break

        # 3. --- HYPER-SPEED OPTIMIZATION: BATCH TAGGING ---
        # Instead of tagging words one-by-one in the loop, we tag the entire pool at once!
        if self.nlp_sp is not None:
            spacy_batch_size = int(params.get("spacy_batch_size", 512))
            self.batch_tag_words(distractor_opts, batch_size=spacy_batch_size)
            
        # 4. Filter the pool using the now-cached high-quality POS tags
        if pos_filter:
            filtered = []
            for w in distractor_opts:
                # Use actual SpaCy POS tag from cache if available
                w_lower = w.lower()
                if hasattr(self, 'pos_cache') and w_lower in self.pos_cache:
                    spacy_pos = self.pos_cache[w_lower]
                    # Convert SpaCy POS to simplified NOUN/!NOUN tag
                    p_tag = "NOUN" if spacy_pos in ('NOUN', 'PROPN') else "!NOUN"
                else:
                    # Fallback to titlecase heuristic if POS not cached
                    is_noun = self.has_titlecase_variant(w)
                    p_tag = "NOUN" if is_noun else "!NOUN"
                
                if pos_filter.startswith('!'):
                    if p_tag == pos_filter[1:]: continue
                elif p_tag != pos_filter:
                    continue
                filtered.append(w)
            distractor_opts = filtered

        random.shuffle(distractor_opts)
        return distractor_opts[:n]



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
            import os
            if not os.path.isabs(exclude) and not os.path.exists(exclude):
                fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), exclude)
                if os.path.exists(fallback):
                    exclude = fallback
            try:
                with open(exclude, "r", encoding="utf-8") as f:
                    exclusions_lower = set(line.strip().lower() for line in f if line.strip())
            except Exception as e:
                import logging
                logging.error(f"Could not load exclude_words from {exclude}: {e}")
                pass

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
        self._build_length_index()


# ---------------------------------------------------------------------------
# German noun suffix heuristic (no spaCy required)
# ---------------------------------------------------------------------------
# Well-known derivational suffixes that almost exclusively form nouns.
_GERMAN_NOUN_SUFFIXES = (
    'ung', 'heit', 'keit', 'schaft', 'tion', 'sion', 'nis', 'tum', 'ling', 
    'tät', 'ment', 'chen', 'lein', 'ismus', 'eur', 'ant', 'ent', 'ist', 'enz', 'anz',
    'ität', 'ik', 'ur', 'ade', 'age', 'ie', 'elle', 'ette', 'ine', 'ive', 'ose',
    'um', 'form', 'werk', 'zeug', 'haus', 'raum', 'platz', 'zeit', 'kraft', 'land',
    # Additional common German endings
    'er', 'el', 'en', 'e' 
)



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
        short_word_min_zipf = float(params.get("short_word_min_zipf", 3.5))

        exclusions_lower = set()
        if exclude is not None:
            import os
            # Try to resolve relative paths against the script directory
            if not os.path.isabs(exclude) and not os.path.exists(exclude):
                fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), exclude)
                if os.path.exists(fallback):
                    exclude = fallback
            try:
                with open(exclude, "r", encoding="utf-8") as f:
                    for line in f:
                        w = line.strip()
                        if w and not w.startswith("#"):
                            exclusions_lower.add(w.lower())
            except Exception as e:
                import logging
                logging.error(f"Could not load exclude_words from {exclude}: {e}")
                pass

        include_words = None
        if include is not None and os.path.exists(include):
            try:
                with open(include, "r", encoding="utf-8") as f:
                    include_words = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            except Exception:
                include_words = None

        freq_dict = wordfreq.get_frequency_dict("de")
        # Merge include_words INTO the main dictionary instead of replacing it.
        # This ensures short curated words are always available as candidates
        # without losing the full wordfreq pool for longer positions.
        source_words = list(freq_dict.keys())
        if include_words is not None:
            source_words = list(include_words) + source_words

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
            if not re.search(r"[aeiouyäöü]", lw):
                continue

            if _is_english_dominant(lw):
                continue
            try:
                z = wordfreq.zipf_frequency(lw, "de")
            except Exception:
                continue
            # Apply length-dependent zipf thresholding to weed out short acronyms
            # but allow rare long compound nouns. 
            # Reverted to len < 5 to ensure short non-nouns have enough fallback candidates.
            effective_min_zipf = min_zipf if len(lw) >= 5 else max(min_zipf, short_word_min_zipf)
            if z < effective_min_zipf:
                continue
            freq_val = z * math.log(10)
            # The word is added; its POS and CASE will be determined 100% by SpaCy on demand
            # in eval_single_word_case, which uses the contextual frame '"Das {word} ist hier."'
            self.words.append(distractor(lw, freq_val, pos=None))
            seen.add(lw)

        # Build case_map: identify which distractor words are German nouns
        # and store their Titlecased form.
        self.case_map = {}
        self.nlp_sp = None
        self._init_spacy()
        self._build_length_index()


    def _init_spacy(self):
        """Load SpaCy model with preference for larger models (better POS accuracy)."""
        try:
            import spacy
            # Try large model first (best accuracy ~96%), then medium, then small
            for model_name in ['de_core_news_lg', 'de_core_news_md', 'de_core_news_sm']:
                try:
                    self.nlp_sp = spacy.load(model_name)
                    logging.info(f"Loaded SpaCy model: {model_name}")
                    return
                except Exception:
                    continue
            self.nlp_sp = None
            logging.warning("No German SpaCy model found. Install with: python -m spacy download de_core_news_lg")
        except Exception:
            self.nlp_sp = None

    def _eval_single_word_case(self, token_lower):
        """Ultimate POS check using standalone SpaCy evaluation."""
        w_cap = token_lower.capitalize()
        
        # For all words: use SpaCy with CAPITALIZED form isolated
        if self.nlp_sp is not None:
            doc_c = self.nlp_sp(w_cap)
            if len(doc_c) > 0 and doc_c[0].pos_ in ('NOUN', 'PROPN'):
                self.case_map[token_lower] = w_cap
            else:
                self.case_map[token_lower] = None
        else:
            self.case_map[token_lower] = None
                
        return self.case_map[token_lower]

    def has_titlecase_variant(self, token):
        """Public check for noun/titlecase status of a token (lowercase or otherwise)."""
        t_lower = token.lower()
        if t_lower not in self.case_map:
            self._eval_single_word_case(t_lower)
        return self.case_map.get(t_lower) is not None

    def get_titlecase_variant(self, token):
        """Returns the TitleCase form if the token is a known noun."""
        t_lower = token.lower()
        if t_lower not in self.case_map:
            self._eval_single_word_case(t_lower)
        return self.case_map.get(t_lower)

    def canonical_case(self, token):
        """Return Titlecased form if known noun, else token as-is."""
        t_lower = token.lower()
        if t_lower not in self.case_map:
            self._eval_single_word_case(t_lower)
        ans = self.case_map.get(t_lower, None)
        return ans if ans is not None else token


def get_frequency_de(word):
    """Returns German Zipf frequency converted to natural-log units."""
    return wordfreq.zipf_frequency(word, 'de') * math.log(10)


def get_frequency_en(word):
    """Returns English Zipf frequency converted to natural-log units."""
    return wordfreq.zipf_frequency(word, 'en') * math.log(10)




def get_thresholds_de(words, params=None):
    """German thresholds based on German frequency.

    When *params* contains ``freq_tolerance`` (in Zipf units), the frequency
    band is tightened to mean_target_freq ± tolerance (converted to natural-log
    units).  This ensures distractors closely match the target word's frequency.
    """
    lengths = []
    freqs = []
    for word in words:
        stripped = strip_punct(word)
        # Clamp word lengths to Boyce-style bins [3, 15] before range creation.
        lengths.append(max(3, min(len(stripped), 15)))
        freqs.append(get_frequency_de(stripped))
    min_length = min(lengths)
    max_length = max(lengths)

    # --- Tight frequency band matching ---
    if params and 'freq_tolerance' in params:
        tol_zipf = float(params['freq_tolerance'])
        tol_natlog = tol_zipf * math.log(10)
        mean_freq = sum(freqs) / len(freqs)
        min_freq = mean_freq - tol_natlog
        max_freq = mean_freq + tol_natlog
    else:
        # Legacy wide-range behavior
        min_freq = min(min(freqs), 11)
        max_freq = max(max(freqs), 3)

    return min_length, max_length, min_freq, max_freq


def get_thresholds_en(words, params=None):
    """English thresholds based on English frequency.

    Supports tight frequency band matching via ``freq_tolerance`` in *params*.
    """
    lengths = []
    freqs = []
    for word in words:
        stripped = strip_punct(word)
        # Clamp word lengths to Boyce-style bins [3, 15] before range creation.
        lengths.append(max(3, min(len(stripped), 15)))
        freqs.append(get_frequency_en(stripped))
    min_length = min(lengths)
    max_length = max(lengths)

    if params and 'freq_tolerance' in params:
        tol_zipf = float(params['freq_tolerance'])
        tol_natlog = tol_zipf * math.log(10)
        mean_freq = sum(freqs) / len(freqs)
        min_freq = mean_freq - tol_natlog
        max_freq = mean_freq + tol_natlog
    else:
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
            import os
            if not os.path.isabs(exclude) and not os.path.exists(exclude):
                fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), exclude)
                if os.path.exists(fallback):
                    exclude = fallback
            try:
                with open(exclude, "r", encoding="utf-8") as f:
                    exclusions_lower = set(
                        strip_arabic_diacritics(line.strip()) for line in f if line.strip()
                    )
            except Exception as e:
                import logging
                logging.error(f"Could not load exclude_words from {exclude}: {e}")
                pass

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
        self._build_length_index()

    def canonical_case(self, token):
        """Arabic has no casing; return as-is."""
        return token

    def get_titlecase_variant(self, token):
        """Arabic has no casing; always returns None."""
        return None


def get_frequency_ar(word):
    """Returns Arabic Zipf frequency converted to natural-log units."""
    return wordfreq.zipf_frequency(strip_arabic_diacritics(word), 'ar') * math.log(10)


def get_thresholds_ar(words, params=None):
    """Arabic thresholds based on Arabic frequency.

    Supports tight frequency band matching via ``freq_tolerance`` in *params*.
    """
    lengths = []
    freqs = []
    for word in words:
        stripped = strip_punct(word)
        # Arabic words can be shorter; clamp to [2, 15].
        lengths.append(max(2, min(len(stripped), 15)))
        freqs.append(get_frequency_ar(stripped))
    min_length = min(lengths)
    max_length = max(lengths)

    if params and 'freq_tolerance' in params:
        tol_zipf = float(params['freq_tolerance'])
        tol_natlog = tol_zipf * math.log(10)
        mean_freq = sum(freqs) / len(freqs)
        min_freq = mean_freq - tol_natlog
        max_freq = mean_freq + tol_natlog
    else:
        min_freq = min(min(freqs), 11)
        max_freq = max(max(freqs), 3)

    return min_length, max_length, min_freq, max_freq

