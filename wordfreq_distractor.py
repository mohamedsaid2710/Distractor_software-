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
        self.nlp_sp = None

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
                if freq_low is not None and freq_high is not None:
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

    def batch_tag_words(self, words, params=None):
        """Tag a list of words in bulk using high-performance Stanza neural tagging.
        
        This method processes words using Stanza's universal POS (UPOS) tagging.
        For German, it identifies nouns and proper nouns to ensure correct TitleCasing,
        leveraging Stanza's deep-learning morphosyntactic analysis.
        """
        if self.nlp_sp is None or not words:
            if self.nlp_sp is None:
                print(f"[DIAG] batch_tag_words SKIPPED: nlp_sp is None", flush=True)
            return

        # Initialize POS cache if not exists
        if not hasattr(self, 'pos_cache'):
            self.pos_cache = {}

        # Filter for words not yet in case_map
        unique_words = list(set(w.lower() for w in words if w.lower() not in self.case_map))
        if not unique_words:
            return

        # Stanza Tagging (German)
        # We pass the raw words to Stanza. Since we are dealing with German morphology
        # Stanza's neural net handles capitalization assumptions internally very well.
        content_words = unique_words
        if not content_words:
            return

        # Read batch size dynamically from params
        batch_size = 2000  # fallback default
        if params is not None:
            try:
                batch_size = int(params.get('nlp_batch_size', params.get('spacy_batch_size', 2000)))
            except Exception:
                pass

        try:
            import stanza
            # Batch process in chunks to prevent memory explosion
            out_docs = []
            display_count = False
            for i in range(0, len(content_words), batch_size):
                chunk = content_words[i:i + batch_size]
                
                # --- GERMAN CONTEXTUAL FRAME FIX ---
                # Tagging 'kiste' isolated -> VERB (incorrect). 
                # Tagging 'Das ist ein Kiste.' -> NOUN (correct).
                if getattr(self, 'lang', 'en') == 'de':
                    if not display_count:
                        print(f"    [NLP] Running Stanza with Sentence Frames on {len(content_words)} candidates...", flush=True)
                        display_count = True
                    in_docs = [stanza.Document([], text=f"Das ist ein {w.capitalize()}.") for w in chunk]
                    out_chunk = self.nlp_sp(in_docs)
                    # Extract result from index 3: "Das"(0) "ist"(1) "ein"(2) "{Word}"(3) "."(4)
                    for word_l, doc in zip(chunk, out_chunk):
                        if doc.sentences and len(doc.sentences[0].words) >= 4:
                            upos = doc.sentences[0].words[3].upos
                        else:
                            upos = 'X'
                        self.pos_cache[word_l] = upos
                        if upos in ('NOUN', 'PROPN'):
                            self.case_map[word_l] = word_l.capitalize()
                        else:
                            self.case_map[word_l] = None
                else:
                    # Non-German or fallback: isolated tagging
                    in_docs = [stanza.Document([], text=w) for w in chunk]
                    out_chunk = self.nlp_sp(in_docs)
                    for word_l, doc in zip(chunk, out_chunk):
                        if doc.sentences and doc.sentences[0].words:
                            upos = doc.sentences[0].words[0].upos
                        else:
                            upos = 'X'
                        self.pos_cache[word_l] = upos
                        if upos in ('NOUN', 'PROPN'):
                            self.case_map[word_l] = word_l.capitalize()
                        else:
                            self.case_map[word_l] = None
        except Exception as e:
            print(f"[DIAG] batch_tag_words Stanza failed: {e}", flush=True)
            for word_l in content_words:
                self.case_map[word_l] = None
                self.pos_cache[word_l] = 'X'

            # Diagnostic: HARD PRINT that bypasses logging config
            noun_count = sum(1 for w in content_words if self.pos_cache.get(w) in ('NOUN', 'PROPN'))
            non_noun_count = len(content_words) - noun_count
            print(f"[DIAG] Batch tagged {len(content_words)} words: {noun_count} NOUN, {non_noun_count} non-NOUN", flush=True)
            noun_examples = [w for w in content_words if self.pos_cache.get(w) in ('NOUN', 'PROPN')][:5]
            non_noun_examples = [f"{w}({self.pos_cache.get(w)})" for w in content_words if self.pos_cache.get(w) not in ('NOUN', 'PROPN')][:5]
            if noun_examples:
                print(f"[DIAG] NOUN examples: {noun_examples}", flush=True)
            if non_noun_examples:
                print(f"[DIAG] Non-NOUN examples: {non_noun_examples}", flush=True)
        except Exception as e:
            logging.error(f"SpaCy batch tagging failed: {e}")
            for word_l in content_words:
                self.case_map[word_l] = None
                self.pos_cache[word_l] = 'X'



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
        if len(distractor_opts) < target_pool_size and min_freq is not None and max_freq is not None:
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
            self.batch_tag_words(distractor_opts, params=params)
        else:
            print(f"[DIAG] SKIPPING batch_tag: nlp_sp is None! pos_cache will be empty.", flush=True)
        # 4. Filter the pool using the now-cached high-quality POS tags
        if pos_filter:
            filtered = []
            for w in distractor_opts:
                w_lower = w.lower()
                # --- DUAL-CHECK: Combine Stanza POS + wordfreq titlecase heuristic ---
                # Stanza can misclassify German nouns when fed isolated lowercase words.
                # wordfreq's titlecase heuristic reflects actual corpus capitalisation.
                # If EITHER source says NOUN, treat as NOUN (conservative approach).
                stanza_says_noun = False
                if hasattr(self, 'pos_cache') and w_lower in self.pos_cache:
                    stanza_says_noun = self.pos_cache[w_lower] in ('NOUN', 'PROPN')

                wordfreq_says_noun = self.has_titlecase_variant(w) if hasattr(self, 'has_titlecase_variant') else False

                is_noun = stanza_says_noun or wordfreq_says_noun
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
        self.nlp_sp = None
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

        # === PRELOAD EN POS CACHE ===
        if not hasattr(self, 'pos_cache'):
            self.pos_cache = {}
        try:
            import json
            import os
            cache_file = "models/english_code/english_pos_cache.json"
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    self.pos_cache.update(cached_data)
                print(f"[CACHE] Successfully loaded {len(cached_data)} POS tags from {cache_file}!", flush=True)
        except Exception as e:
            print(f"[CACHE] Error loading EN POS cache: {e}")

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
        # === PRELOAD NOUN CACHE (NEW) ===
        if not hasattr(self, 'pos_cache'):
            self.pos_cache = {}
        if not hasattr(self, 'case_map'):
            self.case_map = {}
        
        try:
            import json
            import os
            cache_file = "models/german_code/german_pos_cache.json"
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    self.pos_cache.update(cached_data)
                
                for lw, upos in cached_data.items():
                    if upos in ('NOUN', 'PROPN'):
                        self.case_map[lw] = lw.capitalize()
                    else:
                        self.case_map[lw] = None
                        
                print(f"[CACHE] Successfully loaded {len(cached_data)} POS tags from {cache_file}!", flush=True)
        except Exception as e:
            print(f"[CACHE] Error loading POS cache: {e}")
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
        # Track which words came from include_words so they bypass frequency filters
        include_set = set(w.lower().strip() for w in (include_words or []))
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

            if _is_english_dominant(lw):
                continue
            try:
                z = wordfreq.zipf_frequency(lw, "de")
            except Exception:
                continue
            # Include words bypass frequency filter — they are curated
            if lw not in include_set:
                effective_min_zipf = min_zipf if len(lw) >= 5 else max(min_zipf, short_word_min_zipf)
                if z < effective_min_zipf:
                    continue
            freq_val = z * math.log(10)
            # The word is added; its POS and CASE will be determined 100% by SpaCy on demand
            # in eval_single_word_case, which uses the contextual frame '"Das {word} ist hier."'
            self.words.append(distractor(lw, freq_val, pos=None))
            seen.add(lw)

        # (Do not wipe case_map or pos_cache here, they were populated from cache above)
        self.nlp_sp = None
        self._init_spacy()
        self._build_length_index()


    def _init_spacy(self):
        """Load Stanza model for German NLP (replacing legacy SpaCy)."""
        try:
            import stanza
            # Try to load; download if necessary
            use_gpu = True
            try:
                self.nlp_sp = stanza.Pipeline('de', processors='tokenize,mwt,pos,lemma', use_gpu=use_gpu)
            except Exception:
                stanza.download('de')
                self.nlp_sp = stanza.Pipeline('de', processors='tokenize,mwt,pos,lemma', use_gpu=use_gpu)
            logging.info("Loaded Stanza model for DE token classification.")
        except ImportError:
            logging.error("Stanza not found. Please install with: pip install stanza")
            self.nlp_sp = None
        except Exception as e:
            logging.error(f"Error loading Stanza: {e}")
            self.nlp_sp = None

    def _eval_single_word_case(self, token_lower):
        """POS check using Stanza."""
        if not hasattr(self, 'pos_cache'):
            self.pos_cache = {}

        if self.nlp_sp is not None:
            try:
                doc = self.nlp_sp(token_lower)
                if doc.sentences and doc.sentences[0].words:
                    final_pos = doc.sentences[0].words[0].upos
                else:
                    final_pos = 'X'
                
                self.pos_cache[token_lower] = final_pos
                if final_pos in ('NOUN', 'PROPN'):
                    self.case_map[token_lower] = token_lower.capitalize()
                else:
                    self.case_map[token_lower] = None
            except Exception:
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
        self.nlp_sp = None
        self.segmenter = None
        
        # Initialize Farasa for Arabic NLP
        try:
            from farasa.pos import FarasaPOSTagger
            # interactive=True spawns a persistent Java process, making it much faster
            self.nlp_sp = FarasaPOSTagger(interactive=True)
            print("[INFO] Farasa POSTagger loaded successfully for Arabic.", flush=True)
        except Exception as e:
            print(f"[WARN] Farasa POSTagger not found or failed to load. Install 'farasapy' for Arabic POS tagging. Error: {e}")

        try:
            from farasa.segmenter import FarasaSegmenter
            self.segmenter = FarasaSegmenter(interactive=True)
            print("[INFO] Farasa Segmenter loaded successfully for Arabic prefix filtering.", flush=True)
        except Exception as e:
            print(f"[WARN] Farasa Segmenter not found. Arabic distractors may contain 'waow' prefixes. Error: {e}")

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

        # === PRELOAD AR POS CACHE ===
        if not hasattr(self, 'pos_cache'):
            self.pos_cache = {}
        try:
            import json
            import os
            cache_file = "models/arabic_code/arabic_pos_cache.json"
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    self.pos_cache.update(cached_data)
                print(f"[CACHE] Successfully loaded {len(cached_data)} POS tags from {cache_file}!", flush=True)
        except Exception as e:
            print(f"[CACHE] Error loading AR POS cache: {e}")

        include_words = None
        if include is not None and os.path.exists(include):
            try:
                with open(include, "r", encoding="utf-8") as f:
                    include_words = [line.strip() for line in f if line.strip()]
            except Exception:
                include_words = None

        freq_dict = wordfreq.get_frequency_dict("ar")
        source_words = include_words if include_words is not None else freq_dict.keys()

        # Define clitics to filter out from distractor base dictionary
        # This prevents generating distractors with obvious agglutinated prefixes
        banned_clitic_prefixes = ('و+', 'ف+', 'ب+', 'ك+', 'ل+')
        
        self.words = []
        seen = set()
        
        # Batch segment the source words if Segmenter is loaded
        # Farasa Segmenter can be slow one by one, so we pre-segment the most frequent N words
        # but wait, the loop processes raw from iterators, we can just segment per word since
        # freq dict is large. Let's segment on the fly but only for words that pass initial regex.
        
        print("[INFO] Building Arabic distractor vocabulary, isolating clean stems...", flush=True)
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
            
            # Farasa Prefix filter
            if self.segmenter:
                try:
                    seg = self.segmenter.segment(token)
                    # if the segmented token starts with any banned clitic prefix
                    if any(seg.startswith(clitic) for clitic in banned_clitic_prefixes):
                        continue
                except Exception:
                    pass
                
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

    def has_titlecase_variant(self, token):
        """Arabic has no title casing; always returns False."""
        return False

    def batch_tag_words(self, words, params=None):
        """Tag Arabic words using Farasa POSTagger."""
        if getattr(self, 'nlp_sp', None) is None or not words:
            return

        if not hasattr(self, 'pos_cache'):
            self.pos_cache = {}

        unique_words = list(set(w for w in words if w not in self.pos_cache))
        if not unique_words:
            return

        print(f"    [NLP] Running Farasa POS Tagger on {len(unique_words)} Arabic candidates...", flush=True)
        # Farasa interactive doesn't have a reliable batch API method, so run loops. 
        # Interactive mode avoids re-loading the jar for each call so it's quite fast.
        for w in unique_words:
            try:
                # e.g., "S-و/CONJ+ال/DET+قمر/NOUN"
                tagged = self.nlp_sp.tag(w)  # this sometimes has whitespace/newlines
                parts = tagged.split('+')
                pos = 'X'
                # Find the first true lexical tag from the end
                for part in reversed(parts):
                    if '/' in part:
                        curr = part.split('/')[-1].strip().upper()
                        # Filter out common prefix clitic classifications
                        if curr not in ('CONJ', 'PREP', 'DET', 'PART', 'PUNC'):
                            pos = curr
                            break
                if pos == 'X' and parts:
                    if '/' in parts[-1]:
                        pos = parts[-1].split('/')[-1].strip().upper()
                
                # Standardize Farasa outputs to Universal POS (UPOS) tags
                # so it maps correctly with the rest of the software's filters
                if pos == 'V': pos = 'VERB'
                elif pos == 'PREP': pos = 'ADP'
                elif pos == 'CONJ': pos = 'CCONJ'
                elif 'PRON' in pos: pos = 'PRON'
                elif pos not in ('NOUN', 'ADJ', 'ADV', 'NUM', 'DET', 'PART', 'PROPN'):
                    if pos == 'X':
                        pass # keep as X
                    else:
                        pos = 'X' # fallback
                
                self.pos_cache[w] = pos
            except Exception:
                self.pos_cache[w] = 'X'

        # Try to save to cache file just to keep running performance high
        try:
            import json
            cache_file = "models/arabic_code/arabic_pos_cache.json"
            import os
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self.pos_cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


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

