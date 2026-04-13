import wordfreq
import os
import re
import math
import random
import logging
import json
import bisect
from collections import defaultdict

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


# QUALITY GATE: Runtime validator for cache words
def _is_valid_cache_word(word: str, min_target_length: int = 3, lang: str = 'de') -> bool:
    """
    Quality gate for words pulled from POS cache.
    Rejects garbage patterns that pass frequency checks but are corrupted (xte, odr, rav, etc).
    
    Args:
        word: Word to validate
        min_target_length: Minimum target word length (relaxes gates for short words)
        lang: Language code ('de' for German enforcement)
    """
    word_lower = word.lower()
    word_len = len(word_lower)
    
    # Gate 1: Minimum structure - proportional to target length
    # For 3-char targets, accept 3-char candidates
    # For 4-5 char targets, accept 4+ char candidates  
    # For 6+ char targets, enforce stricter minimums
    min_word_len = max(2, min_target_length)
    if word_len < min_word_len:
        return False
    
    # Gate 2: Must have German vowels
    if not re.search(r'[aeioäöüy]', word_lower):
        return False
    
    # Gate 3: Cannot be mostly same character (catches "aaaaaa", "bbbb", etc)
    # STRICTER for short words (they have more repetition naturally)
    char_freq = {}
    for c in word_lower:
        char_freq[c] = char_freq.get(c, 0) + 1
    max_freq = max(char_freq.values())
    
    # Short words (<4 chars): max 60% same char (allows "aaa", "bbb")
    # Medium words (4-7): max 50% same char
    # Long words (8+): max 40% same char
    if word_len < 4:
        threshold = 0.6
    elif word_len < 8:
        threshold = 0.5
    else:
        threshold = 0.4
    
    if max_freq / word_len > threshold:
        return False
    
    # Gate 4: Must have consonants (not pure vowels)
    # RELAXED for very short words (2-3 chars can be vowel-heavy)
    if word_len >= 4:
        if not re.search(r'[bcdfghjklmnpqrstvwxz]', word_lower):
            return False
    
    # Gate 5: CRITICAL ZIPF VALIDATION (catches 'xte' Zipf:2.04, 'odr' Zipf:2.55, etc)
    # For German, enforce strict minimum Zipf for short words
    # Short words (≤5 chars) must have Zipf ≥ 4.0 (very confident)
    # Medium words (6-8 chars) must have Zipf ≥ 3.5 (confident)
    # Long words (9+ chars) can relax to Zipf ≥ 3.0 (wordfreq may not know compounds)
    if lang == 'de' and word_len < 9:
        try:
            z = wordfreq.zipf_frequency(word_lower, 'de')
            if word_len <= 5:
                if z < 4.0:  # STRICT for short words
                    return False
            elif word_len <= 8:
                if z < 3.5:  # MEDIUM for medium words
                    return False
        except Exception:
            return False  # If wordfreq fails, reject the word
    
    return True


class wordfreq_dict(distractor_dict):
    """General class of dictionaries"""

    def __init__(self, params={}):
        self.params = params
        self.words = []
        self.words_by_len = {}
        self.nouns_by_len = defaultdict(set)
        self.others_by_len = defaultdict(set)
        self.lang = None
        self.nlp_sp = None

    def get_words_by_len(self, desired_len):
        """Standard accessor for all words of a certain length."""
        pool = self.words_by_len.get(desired_len, [])
        # Return just the text for compatibility with old Stage 2 logic if still used
        return [w.text for w in pool]

    def get_best_frequency_pool(self, desired_len, target_freq, n=400):
        """
        Retrieves a 'neighborhood' of words with frequencies closest to target_freq.
        Uses binary search for O(log N) neighborhood identification.
        """
        full_pool = self.words_by_len.get(desired_len, [])
        if not full_pool:
            return []
            
        if len(full_pool) <= n:
            return [w.text for w in full_pool]
            
        # Since full_pool is sorted by frequency DESCENDING, we use a custom key
        # frequencies are [7.0, 6.9, ..., 1.0]
        # To find 'target_freq' in a descending list, we can search in the negative list
        # or just use bisect with a custom order.
        
        # Extract Zipf frequencies for binary search
        # NOTE: Using negative frequencies so bisection on descending order works
        freqs_neg = [-w.freq for w in full_pool] # [-7.0, -6.9, ..., -1.0] (ascending)
        
        idx = bisect.bisect_left(freqs_neg, -target_freq)
        
        # Take n/2 words from each side of the index
        start = max(0, idx - (n // 2))
        end = min(len(full_pool), start + n)
        
        # Adjust start if we hit the end
        if end == len(full_pool):
            start = max(0, end - n)
            
        neighborhood = full_pool[start:end]
        return [w.text for w in neighborhood]

    def _build_length_index(self):
        """Internal helper to organize words by length for frequency-matched fallback lookups."""
        self.words_by_len = {}
        for w in self.words:
            # Categorize by length
            l = getattr(w, 'len', len(strip_punct(w.text)))
            if l not in self.words_by_len:
                self.words_by_len[l] = []
            self.words_by_len[l].append(w)
        
        # KEY FIX: Sort each bucket by frequency (descending) 
        # to allow fast searching for closest frequency matches.
        for l in self.words_by_len:
            self.words_by_len[l].sort(key=lambda x: getattr(x, 'freq', 0), reverse=True)
        
        # 2. Categorized indices using the POS CACHE (if available)
        pos_cache = getattr(self, 'pos_cache', {})
        for lw, category in pos_cache.items():
            # QUALITY GATE: Skip garbage patterns
            # Validate each word relative to its own length
            _lang = getattr(self, 'lang', 'de')  # Default to German for legacy code
            if not _is_valid_cache_word(lw, min_target_length=len(lw), lang=_lang):
                continue
                
            l = len(lw)
            # TIGHTENED: A word is a noun if its category says so, OR if it's already in the noun pool
            # This prevents lowercased nouns from leaking into others_by_len
            if category in ('NOUN', 'PROPN'):
                self.nouns_by_len[l].add(lw)
            else:
                # Double-check: if it has been marked as a noun elsewhere, don't put in 'others'
                if lw not in self.nouns_by_len[l]:
                    self.others_by_len[l].add(lw)

    def get_emergency_pool(self, length, is_noun=False):
        """Returns a pre-indexed list of words for the specified length and category."""
        if is_noun:
            return list(self.nouns_by_len.get(length, []))
        return list(self.others_by_len.get(length, []))

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
        """Tag a list of words in bulk using Stanza neural tagging. 
        
        Simple isolated word tagging (no context frames) to preserve pure two-mode logic.
        """
        if self.nlp_sp is None or not words:
            if self.nlp_sp is None:
                print(f"[DIAG] batch_tag_words SKIPPED: nlp_sp is None", flush=True)
            return

        if not hasattr(self, 'pos_cache'):
            self.pos_cache = {}

        # ABSOLUTE TRUTH GUARD: Only tag words that are NOT already in our verified cache.
        unique_words = list(set(w.lower() for w in words if w.lower() not in self.pos_cache))
        
        if not unique_words:
            # print(f"    [NLP] Skipping Stanza: All {len(words)} candidates are already verified in Cache.", flush=True)
            return

        batch_size = 256
        if params is not None:
            try:
                batch_size = int(params.get('nlp_batch_size', params.get('spacy_batch_size', 256)))
            except Exception:
                pass

        try:
            import stanza
            display_count = False
            
            for i in range(0, len(unique_words), batch_size):
                chunk = unique_words[i:i + batch_size]
                
                if not display_count:
                    print(f"    [NLP] Running Stanza (isolated tagging) on {len(unique_words)} candidates...", flush=True)
                    display_count = True
                
                # German Specific Stanza Logic (Case Preservation + Framing is CRITICAL)
                if getattr(self, 'lang', None) == 'de':
                    # WRAP candidates in a neutral frame for context-sensitive tagging.
                    # This ensures words like 'Ausschau' are tagged as NOUN, not ADJ.
                    framed_texts = [f"Das ist {w}." for w in chunk]
                    in_docs = [stanza.Document([], text=t) for t in framed_texts]
                    out_chunk = self.nlp_sp(in_docs)
                    
                    for word_raw, doc in zip(chunk, out_chunk):
                        # The candidate word is the 3rd token in "Das ist [WORD]."
                        if doc.sentences and len(doc.sentences[0].words) >= 3:
                            upos = doc.sentences[0].words[2].upos
                        else:
                            upos = 'X'
                        
                        # FINAL NACIG GUARD: 
                        # If our heuristics are 100% sure it's a NOUN, we override Stanza's last word.
                        common = self.common_casing.get(word_raw, "")
                        
                        # Only apply noun suffix override if the word is capitalized or known to be a noun
                        is_cap = (word_raw and word_raw[0].isupper()) or (common and common[0].isupper())
                        if is_cap and word_raw.endswith(self.NOUN_SUFFIXES):
                            upos = "NOUN"
                        elif (common and common[0].isupper()):
                            upos = "NOUN"
                        
                        # ABSOLUTE TRUTH: Once in Cache, it is immutable. 
                        # We don't override existing cache entries with Stanza hallucinations.
                        word_l = word_raw.lower()
                        if word_l not in self.pos_cache or self.pos_cache[word_l] in ('X', 'UNKNOWN'):
                            self.pos_cache[word_l] = upos
                        
                        # Map casing in JSON cache
                        if self.pos_cache[word_l] in ('NOUN', 'PROPN'):
                            self.case_map[word_l] = word_raw.capitalize()
                        else:
                            self.case_map[word_l] = None
                else:
                    # Non-German: standard isolated tagging
                    in_docs = [stanza.Document([], text=w) for w in chunk]
                    out_chunk = self.nlp_sp(in_docs)
                    for word_l, doc in zip(chunk, out_chunk):
                        if doc.sentences and doc.sentences[0].words:
                            upos = doc.sentences[0].words[0].upos
                        else:
                            upos = 'X'
                        self.pos_cache[word_l] = upos
                        self.case_map[word_l] = word_l.capitalize() if upos in ('NOUN', 'PROPN') else None
                    
        except Exception as e:
            print(f"[DIAG] batch_tag_words Stanza failed: {e}", flush=True)
            for word_l in unique_words:
                self.case_map[word_l] = None
                self.pos_cache[word_l] = 'X'



    def get_potential_distractors(self, min_length, max_length, min_freq, max_freq, params, pos_filter=None):
        """Returns list of candidates, using heuristic first, then widening, then batch SpaCy validation."""
        _lang = getattr(self, 'lang', 'de')
        n = params.get('num_to_test', 200)
        # Fetch MORE so that after POS filtering we still have 'n' candidates.
        # If we have a POS filter, we need a significantly larger pool to find survivors.
        target_pool_size = max(n * 2, 500)
        if pos_filter:
            target_pool_size = max(n * 5, 2000)
        
        # Get exclude list from params for pre-filtering
        exclude_words_set = set()
        exclude_path = params.get('exclude_words', None)
        if exclude_path:
            import os
            # Resolve relative paths
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
        
        # TIER 1: Frequency-Matched Fetch (Status Quo)
        distractor_opts = self.get_words(min_length, max_length, min_freq, max_freq, pos_filter=None, use_spacy=False)
        
        # SHORT-WORD PROTECTION: For ≤5 char words, apply QUALITY GATE
        # (Protects against web-scraping junk like 'fug', 'xte', 'uru' that enter via wordfreq)
        # NOTE: AIR-GAP OPTIMIZATION - Do NOT require words to be in JSON cache.
        # Instead, rely on _is_valid_cache_word() quality gates (Zipf, vowels, etc.)
        # Pre-batch-tagging happens later in the pipeline if needed.
        if max_length <= 5:
            distractor_opts = [w for w in distractor_opts if _is_valid_cache_word(strip_punct(w).lower(), min_target_length=min_length, lang=_lang)]
        
        # PRE-FILTER: Remove excluded words
        if exclude_words_set:
            distractor_opts = [w for w in distractor_opts if strip_punct(w).lower() not in exclude_words_set]
        
        # TIER 1.5: Adaptive Frequency Widening
        # If tight band failed, use the optimized neighborhood search (binary search)
        if len(distractor_opts) < target_pool_size and 'target_zipf' in params:
            target_zipf = params['target_zipf']
            logging.info(f"Tight band search starved. Using neighborhood search around Zipf {target_zipf:.2f}")
            
            # Use binary-search-based neighborhood identifier
            # Start with a generous neighborhood (n=1000) for fast retrieval
            target_exact_len = params.get('target_exact_length') or ((min_length + max_length) // 2)
            # Try exact length neighborhood first
            neighbor_pool = self.get_best_frequency_pool(target_exact_len, target_zipf, n=1000)
            
            # Filter and add to candidates
            if exclude_words_set:
                neighbor_pool = [w for w in neighbor_pool if w not in exclude_words_set]
            
            distractor_opts.extend(neighbor_pool)
            distractor_opts = list(set(distractor_opts))

        # TIER 2: Quality Gate Fallback
        # If still starving, we can pull from other nearby lengths using the same frequency search
        if len(distractor_opts) < target_pool_size:
            target_exact_len = params.get('target_exact_length') or ((min_length + max_length) // 2)
            target_zipf = params.get('target_zipf', 5.0)
            for length_diff in [1, -1]: # Try nearby length neighborhoods
                adj_len = target_exact_len + length_diff
                if adj_len >= 3:
                    adj_pool = self.get_best_frequency_pool(adj_len, target_zipf, n=500)
                    if exclude_words_set:
                        adj_pool = [w for w in adj_pool if w not in exclude_words_set]
                    distractor_opts.extend(adj_pool)
                    distractor_opts = list(set(distractor_opts))
                if len(distractor_opts) >= target_pool_size:
                    break

        # TIER 3: GRADUAL EMERGENCY EXPANSION (Ultimate Quality fallback)
        # If thresholds are NOT met, gradually expand length tolerance.
        if len(distractor_opts) < target_pool_size and hasattr(self, 'get_emergency_pool'):
            target_exact_len = params.get('target_exact_length') or ((min_length + max_length) // 2)
            
            # Expansion Tiers: +/- 1, then +/- 2 (if word is long enough)
            expansion_scales = [1]
            if target_exact_len >= 8:
                expansion_scales.append(2)
            
            for scale in expansion_scales:
                for diff in range(1, scale + 1):
                    lengths_to_try = [target_exact_len + diff, target_exact_len - diff]
                    for l in lengths_to_try:
                        if l < 2: continue # Never go below 2
                        # Use target_is_noun from params for faster pre-filtered expansion
                        target_is_noun_p = params.get('target_is_noun', False)
                        
                        if pos_filter == 'NOUN' or (pos_filter is None and target_is_noun_p):
                            pool = self.get_emergency_pool(l, is_noun=True)
                        elif pos_filter == '!NOUN' or (pos_filter is None and not target_is_noun_p):
                            pool = self.get_emergency_pool(l, is_noun=False)
                        else:
                            pool = self.get_emergency_pool(l, is_noun=True) + self.get_emergency_pool(l, is_noun=False)
                        
                        distractor_opts.extend(pool)
                        distractor_opts = list(set(distractor_opts))
                    
                    if len(distractor_opts) >= target_pool_size:
                        break
                if len(distractor_opts) >= target_pool_size:
                    break

        # 3. --- HYPER-SPEED OPTIMIZATION: BATCH TAGGING ---
        if self.nlp_sp is not None:
            self.batch_tag_words(distractor_opts, params=params)
        
        # Pre-compute quality gate params once (German only)

        # 4. FINAL FILTER: Apply Ironclad Casing, Noise, and PROPN checks
        # [German Cache Injection]: Ensure ANY leftover untagged words get a heuristic pass 
        # before they hit the Grammar Guard.
        
        match_casing_only = params.get('match_casing_only', False)
        target_is_capitalized = params.get('target_is_capitalized', False)
        exclude_propn = params.get('exclude_propn_candidates', False)
        
        _min_val = float(params.get('min_zipf', 1.5))
        _min_json_zipf = float(params.get('json_min_zipf', _min_val))
        _german_vowels = re.compile(r'[aeiouyäöü]', re.IGNORECASE)

        filtered = []
        for w in distractor_opts:
            # --- NOISE & QUALITY FILTER ---
            if not any(c.isalpha() for c in w) or len(w) < 1:
                continue
            if '-' in w and all(len(part) < 2 for part in w.split('-')): # kills x-x-x
                continue

            # --- GERMAN WORD QUALITY GATE ---
            # Kills garbage fragments (xte, fug, uru, rüf, ryl, gge…) that
            # enter via TIER 2/3 JSON fallback without any frequency check.
            # Long compounds (≥ 8 chars) are exempt: wordfreq may not know them.
            if _lang == 'de':
                w_body = strip_punct(w).lower()
                
                # LEXICAL GARBAGE FILTER (IRONCLAD)
                # Removes 'ua', 'xy', 'uv', 'og', 'uefa', etc.
                if _is_lexically_garbage(w_body, 'de'):
                    continue

                if len(w_body) < 8:
                    # Must contain at least one German vowel
                    if not _german_vowels.search(w_body):
                        continue
                    # Must have a real German wordfreq score
                    if wordfreq.zipf_frequency(w_body, 'de') < _min_json_zipf:
                        continue
                    # Reject words more frequent in English than German (e.g. 'Namespace')
                    if _is_english_dominant(w_body):
                        continue

            # --- IRONCLAD CASING ---
            if match_casing_only:
                # [German Exception]: The dictionary is lowercase, so we bypass this check
                # and let the Grammar Guard handle it downstream via POS tagging.
                if _lang != 'de':
                    # If target is capitalized, distractor MUST be capitalized
                    if target_is_capitalized and not w[0].isupper():
                        continue
                    # If target is lowercase, distractor MUST be lowercase
                    if not target_is_capitalized and not w[0].islower():
                        continue

            # --- POS & PROPN REJECTION (Supreme Authority) ---
            w_lower = w.lower()
            in_cache = hasattr(self, 'pos_cache') and w_lower in self.pos_cache
            
            is_propn = False
            is_noun = False
            
            if in_cache:
                tag = self.pos_cache[w_lower]
                if tag == 'X':
                    continue # Ironclad X Rejection
                is_propn = (tag == 'PROPN')
                is_noun = (tag in ('NOUN', 'PROPN'))
            
            # Generic fallback if not already determined in cache
            if not is_noun and not in_cache:
                # Use titlecase variant check for ANY language, not just non-German
                is_noun = self.has_titlecase_variant(w) if hasattr(self, 'has_titlecase_variant') else False

            # 1. Reject Proper Nouns (the "rubbish" fragments like Obb, Dha)
            if exclude_propn and is_propn:
                continue

            # 2. GERMAN GRAMMAR GUARD (Inviolable)
            # Use target_is_noun from params for maximum precision (handles sentence-starts).
            # If target is a noun, distractor must be a noun.
            # If target is NOT a noun (even if capitalized at start), distractor must NOT be a noun.
            if _lang == 'de':
                target_is_noun_pos = params.get('target_is_noun', target_is_capitalized)
                
                # REJECTION 1: Noun used for Non-Noun target (prevents giveaways like 'das' -> 'Merz')
                if not target_is_noun_pos and is_noun:
                    continue
                
                # REJECTION 2: Non-Noun used for Noun target (mandatory category alignment)
                if target_is_noun_pos and not is_noun:
                    continue

                # Apply dictionary's native casing to properly check for lowercase violations
                # Nouns in German MUST be capitalized. If they remain lowercase even after mapping, reject.
                canonical_w = self.case_map.get(w, None) if hasattr(self, 'case_map') else None
                native_w = canonical_w if canonical_w else w

                # REJECTION 3: Noun-candidate is NATIVELY LOWERCASE (Fatal giveaway in German)
                if is_noun and native_w[0].islower():
                    continue

            filtered.append(w)
        distractor_opts = filtered

        # 5. LIMIT SIZE (Return Target Pool)
        random.shuffle(distractor_opts)
        
        # --- CASING RE-ENFORCEMENT ---
        # Ensure returned strings have correct casing so downstream sentence_set heuristics work on untagged words!
        if _lang == 'de' and target_is_capitalized:
            distractor_opts = [w.capitalize() for w in distractor_opts]

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
        super().__init__(params)
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
    # NACIG Protocol: Ironclad German Noun Suffixes
    NOUN_SUFFIXES = (
        "ung", "heit", "keit", "schaft", "tion", "sion", 
        "tät", "ismus", "ment", "ik", "anz", "enz",
        "eur", "ling", "ist", "or", "erich", "in", "innen"
    )

    """Zipf-based German dictionary built from the wordfreq library."""

    def __init__(self, params={}):
        super().__init__(params)
        self.lang = "de"
        exclude = params.get("exclude_words", "exclude_de.txt")
        min_word_len = int(params.get("min_word_len", 3))
        min_zipf = float(params.get("min_zipf", 3.0))
        short_word_min_zipf = float(params.get("short_word_min_zipf", 3.5))

        exclusions_lower = set()
        if exclude is not None:
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
                logging.error(f"Could not load exclude_words from {exclude}: {e}")

        if not hasattr(self, 'pos_cache'):
            self.pos_cache = {}
        if not hasattr(self, 'case_map'):
            self.case_map = {}
        if not hasattr(self, 'overrides'):
            self.overrides = {}
        if not hasattr(self, 'common_casing'):
            self.common_casing = {}

        # === PRELOAD GERMAN POS CACHE ===
        try:
            cache_file = "models/german_code/german_pos_cache_v2.json"
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    self.pos_cache.update(cached_data)
                print(f"    [CACHE] Loaded {len(cached_data)} German POS tags from disk.", flush=True)
                # Seed nouns_by_len and others_by_len pools with loaded data
                self._populate_pools_from_cache()
        except Exception as e:
            logging.error(f"[CACHE] Error loading German POS cache: {e}")

        # Grounded Noun Guard: using spacy's German model for robust orthography
        import spacy
        try:
            # Using 'lg' (Large) model for better accuracy if available
            self.spacy_nlp = spacy.load("de_core_news_lg")
        except:
            # Fallback to 'sm' (Small) if only that is installed
            try:
                self.spacy_nlp = spacy.load("de_core_news_sm")
            except:
                self.spacy_nlp = None

        freq_dict = wordfreq.get_frequency_dict("de")
        source_words = list(freq_dict.keys())

        self.words = []
        seen = set()
        for raw in source_words:
            lw = raw.strip().lower()
            if lw in seen or lw in exclusions_lower:
                continue
            if not re.match(r"^[a-zäöüß]+$", lw):
                continue
            if len(lw) < min_word_len or not re.search(r"[aeiouyäöü]", lw):
                continue
            if _is_english_dominant(lw):
                continue
            try:
                z = wordfreq.zipf_frequency(lw, "de")
            except Exception:
                continue
            
            effective_min_zipf = min_zipf if len(lw) >= 5 else max(min_zipf, short_word_min_zipf)
            if z < effective_min_zipf:
                continue
            freq_val = z * math.log(10)
            self.words.append(distractor(lw, freq_val, pos=None))
            seen.add(lw)
            # Store the common casing as a hint for the POS tagger
            # If the common casing is Titlecase, it's almost certainly a NOUN.
            self.common_casing[lw] = raw.strip()

        self.nlp_sp = None
        self._init_spacy()
        self._build_length_index()
        self._load_pos_overrides(params.get('pos_overrides', 'pos_overrides.txt'))

    def _populate_pools_from_cache(self):
        """Populates nouns_by_len and others_by_len from the loaded pos_cache."""
        if not self.pos_cache:
            return
        
        for word_l, upos in self.pos_cache.items():
            l = len(word_l)
            if not (1 < l < 100): continue # Bounds check
            
            # 3. ABSOLUTE TRUTH: The Cache is final.
            # We trust the UPOS in the JSON file above all else.
            if upos in ('NOUN', 'PROPN'):
                self.case_map[word_l] = word_l.capitalize()
                self.nouns_by_len[l].add(word_l)
                if word_l in self.others_by_len[l]:
                    self.others_by_len[l].remove(word_l)
            else:
                self.case_map[word_l] = None
                self.others_by_len[l].add(word_l)
                if word_l in self.nouns_by_len[l]:
                    self.nouns_by_len[l].remove(word_l)


    def _load_pos_overrides(self, path):
        """Loads manual POS overrides from a file (format: 'word POS')."""
        if not path: return
        if not os.path.exists(path):
            logging.info(f"No POS overrides file found at {path}. Skipping.")
            return
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    parts = line.split()
                    if len(parts) >= 2:
                        word = parts[0].strip().lower()
                        pos = parts[1].strip().upper()
                        self.overrides[word] = pos
            logging.info(f"Loaded {len(self.overrides)} POS overrides from {path}.")
        except Exception as e:
            logging.error(f"Failed to load POS overrides: {e}")

    def _init_spacy(self):
        """Load Stanza model for German NLP (replacing legacy SpaCy)."""
        try:
            import stanza
            # Try to load; download if necessary
            stanza.download('de')
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
                # USE NEUTRAL SENTENCE FRAME FOR 100% ACCURACY
                # "Das ist ein {word}." forces Stanza to see the word in a noun slot.
                # Verbs like 'zog' will be correctly tagged as VERB because they fail the noun slot.
                frame = f"Das ist ein {token_lower}."
                doc = self.nlp_sp(frame)
                
                # The word we care about is at index 3: "Das(0) ist(1) ein(2) {token}(3) .(4)"
                if len(doc.sentences) > 0 and len(doc.sentences[0].words) >= 4:
                    final_pos = doc.sentences[0].words[3].upos
                elif doc.sentences and doc.sentences[0].words:
                    # Fallback to first word if frame failed
                    final_pos = doc.sentences[0].words[0].upos
                else:
                    final_pos = 'X'
                
                self._record_pos_tag(token_lower, final_pos)
            except Exception as e:
                logging.error(f"[STANZA] Frame tagging error for {token_lower}: {e}")
                self.case_map[token_lower] = None
        else:
            self.case_map[token_lower] = None

        return self.case_map[token_lower]

    def has_titlecase_variant(self, token):
        """Public check for noun/titlecase status of a token (lowercase or otherwise).
        
        Strategy: Use runtime-tagged POS cache (from batch_tag_words) which has correct Stanza tags.
        Falls back to titlecase heuristic if not yet tagged.
        Skips the pre-loaded 634k JSON cache (which may be corrupted).
        """
        t_lower = token.lower()
        
        # PRIORITY 0: Manual Overrides
        if hasattr(self, 'overrides') and t_lower in self.overrides:
            return self.overrides[t_lower] in ('NOUN', 'PROPN')

        # PRIORITY 1: Check RUNTIME POS cache (populated by batch_tag_words with correct Stanza)
        if hasattr(self, 'pos_cache') and t_lower in self.pos_cache:
            pos_tag = self.pos_cache[t_lower]
            result = pos_tag in ('NOUN', 'PROPN')
            return result
        
        # PRIORITY 2: Check JSON Dictionary Pre-filtered Noun List (Critical for 0-Candidate Fallback pools)
        if getattr(self, 'lang', None) == 'de' and hasattr(self, 'nouns_by_len'):
            if t_lower in self.nouns_by_len.get(len(t_lower), set()):
                return True

        # PRIORITY 3: Fallback to titlecase heuristic (Suffix-based ONLY)
        # We NO LONGER trigger neural _eval_single_word_case for the German Absolute Truth phase.
        # Unknown words are treated as Non-Nouns unless they have a classic noun suffix.
        if t_lower.endswith(self.NOUN_SUFFIXES):
            return True
            
        return False

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

    def batch_tag_words(self, words, params=None, force_refresh=False):
        """Batch-tag German candidates using Stanza with SMART SELECTIVE FRAMING.
        
        Categorizes words by morphology and uses context-appropriate sentences:
        - Verbs (6+ chars ending -te, -ten, -iert, -et): Frame "Er {word} es."
        - Adjectives (-lich, -ig, -isch, -keit): Frame "Das ist {word}."
        - Nouns (capitalized or -e, -in, -tion, -heit): Frame "Das ist ein {word}."
        - Others: Isolated tagging
        """
        if self.nlp_sp is None or not words:
            return

        if not hasattr(self, 'pos_cache'):
            self.pos_cache = {}

        # Identify words that need tagging
        if force_refresh:
            to_tag = list(set(w.lower() for w in words))
            print(f"    [STANZA] Force-verifying {len(to_tag)} candidates to purge messy cache...", flush=True)
        else:
            to_tag = []
            for w in set(words):
                w_lower = w.lower()
                w_clean = re.sub(r"[^\w\säöüÄÖÜß]", "", w_lower).strip()
                if not w_clean: continue
                
                # STRICT CACHE TRUTH: If it exists in cache (even as 'X'), do NOT re-tag.
                if w_clean not in self.pos_cache:
                    to_tag.append(w_clean)
            
            to_tag = list(set(to_tag))
            if not to_tag:
                return
            
            # (Air-gap limit removed. Run Stanza on all out-of-cache candidate pools to ensure POS accuracy.)
            

        BATCH_SIZE = int(params.get('nlp_batch_size', 256)) if params else 256
        
        # Performance Upgrade: Batch the SpaCy pre-check for the ENTIRE list at once
        spacy_noun_map = {}
        if self.spacy_nlp and to_tag:
            try:
                # Use nlp.pipe for massively parallel SpaCy processing
                # FIX: Capitalize the words so SpaCy accurately detects nouns
                capitalized_to_tag = [w.capitalize() for w in to_tag]
                for doc in self.spacy_nlp.pipe(capitalized_to_tag, batch_size=BATCH_SIZE):
                    if len(doc) > 0 and doc[0].pos_ in ['NOUN', 'PROPN']:
                        spacy_noun_map[doc.text.lower()] = True
            except Exception as e:
                logging.warning(f"[SPACY] Batched pre-check failed: {e}")

        for i in range(0, len(to_tag), BATCH_SIZE):
            batch = to_tag[i : i + BATCH_SIZE]
            
            # Categorize by morphology
            verb_words = []
            adj_words = []
            noun_words = []
            other_words = []
            
            for w in batch:
                w_lower = w.lower()
                common = self.common_casing.get(w_lower, "")
                is_ironclad = (common and common[0].isupper()) or w_lower.endswith(self.NOUN_SUFFIXES)
                
                # Check our pre-calculated SpaCy map
                is_grounded_noun = is_ironclad or spacy_noun_map.get(w_lower, False)
                
                # VERB: 6+ chars ending in past tense markers
                is_verb = (len(w_lower) >= 6 and re.search(r'(te|ten|iert|et)$', w_lower))
                
                if is_grounded_noun:
                    noun_words.append(w_lower)
                elif is_verb:
                    verb_words.append(w_lower)
                elif re.search(r'(lich|ig|isch|keit)$', w_lower):
                    adj_words.append(w_lower)
                elif w[0].isupper() or re.search(r'(tion|heit|ness|mann|schaft|um)$', w_lower):
                    noun_words.append(w_lower)
                else:
                    other_words.append(w_lower)
            
            try:
                import stanza
                # 1. VERB FRAME: "Er {word} es."
                if verb_words:
                    frames = [f"Er {w} es." for w in verb_words]
                    docs = self.nlp_sp([stanza.Document([], text=t) for t in frames])
                    for word, doc in zip(verb_words, docs):
                        if doc.sentences and len(doc.sentences[0].words) >= 2:
                            final_pos = doc.sentences[0].words[1].upos
                        else:
                            final_pos = 'X'
                        self._record_pos_tag(word, final_pos)
                
                # 2. ADJECTIVE FRAME: "Das ist {word}."
                if adj_words:
                    frames = [f"Das ist {w}." for w in adj_words]
                    docs = self.nlp_sp([stanza.Document([], text=t) for t in frames])
                    for word, doc in zip(adj_words, docs):
                        if doc.sentences and len(doc.sentences[0].words) >= 3:
                            final_pos = doc.sentences[0].words[2].upos
                        else:
                            final_pos = 'X'
                        self._record_pos_tag(word, final_pos)
                
                # 3. NOUN FRAME: "Das ist ein {word}."
                if noun_words:
                    # We capitalize the word in the frame to help Stanza recognize the noun
                    frames = [f"Das ist ein {w.capitalize()}." for w in noun_words]
                    docs = self.nlp_sp([stanza.Document([], text=t) for t in frames])
                    for word, doc in zip(noun_words, docs):
                        if doc.sentences and len(doc.sentences[0].words) >= 4:
                            final_pos = doc.sentences[0].words[3].upos
                        else:
                            final_pos = 'X'
                        self._record_pos_tag(word, final_pos)
                
                # 4. ISOLATED: No frame
                if other_words:
                    docs = self.nlp_sp([stanza.Document([], text=w) for w in other_words])
                    for word, doc in zip(other_words, docs):
                        if doc.sentences and doc.sentences[0].words:
                            final_pos = doc.sentences[0].words[0].upos
                        else:
                            final_pos = 'X'
                        self._record_pos_tag(word, final_pos)
                        
            except Exception as e:
                logging.error(f"[STANZA] Max-Throttle batching failed: {e}")
                for w in batch:
                    self._eval_single_word_case(w)

        # Persist to disk
        self.save_pos_cache()
    
    def _record_pos_tag(self, word, upos):
        """Cache-aware POS tagging that populates length-indexed pools.
        TIGHTENED: Ensures nouns never enter the 'others' pool even if tagged with 
        non-noun POS when lowercased.
        """
        if not word: return
        self.pos_cache[word] = upos
        l = len(word)
        
        is_noun = upos in ('NOUN', 'PROPN')
        
        if is_noun:
            self.case_map[word] = word.capitalize()
            self.nouns_by_len[l].add(word)
            # Remove from 'others' if it was mistakenly added before
            if word in self.others_by_len[l]:
                self.others_by_len[l].remove(word)
        else:
            # If not a noun, only add to 'others' if not already known to be a noun
            if word not in self.nouns_by_len[l]:
                self.case_map[word] = None
                self.others_by_len[l].add(word)

    def save_pos_cache(self):
        """Append-only disk write. If a word is already in the cache on disk, do NOT overwrite it!"""
        import json
        import os
        cache_file = "models/german_code/german_pos_cache_v2.json"
        
        # Determine appropriate language cache file
        if getattr(self, 'lang', None) == 'en':
            cache_file = "models/english_code/english_pos_cache.json"
            
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        try:
            existing_cache = {}
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    existing_cache = json.load(f)
            
            # Count how many new keys we add
            added_count = 0
            for k, v in self.pos_cache.items():
                if k not in existing_cache:
                    existing_cache[k] = v
                    added_count += 1
            
            if added_count > 0:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(existing_cache, f, ensure_ascii=False, indent=2)
                # Keep memory perfectly in sync with the canonical disk truth
                self.pos_cache = existing_cache
                print(f"    [CACHE] ADDED {added_count} new POS tags safely to {cache_file} (No Overwrites).")
                
        except Exception as e:
            import logging
            logging.error(f"Failed to safely append POS cache: {e}")


def _is_lexically_garbage(word, lang='de'):
    """Ironclad filter for Non-Lexical Noise.
    
    Rejects:
    - Words with no vowels (if < 6 chars)
    - Words with digits or special symbols (mixed with alphanumeric)
    - Known garbage stems or fragments ('ua', 'xy', 'uv', 'tp', 'og', 'ogm', 'uefa')
    """
    if not word: return True
    w = word.lower()
    
    # 1. Blacklist for known noisy fragments and acronyms from frequency dicts
    blacklist = {
        'ua', 'uv', 'xy', 'tp', 'og', 'ogm', 'uefa', 'usw', 'zb', 'dpa', 'mrd', 'uvm', 'bzw',
        'ops', 'raf', 'lan', 'ole', 'ott', 'ut', 'it', 'ed', 'ex', 're', 'id'
    }
    if w in blacklist:
        return True
        
    # 2. Non-Alpha detection (kills 09, 2024, v2.0, etc.)
    if not w.isalpha():
        # Allow only simple single hyphens for compounds
        if '-' in w:
            parts = w.split('-')
            if any(not p.isalpha() for p in parts):
                return True
        else:
            return True
            
    # 3. German Quality Guard (Density and Structure)
    if lang == 'de':
        vowels = set('aeiouyäöü')
        v_count = sum(1 for c in w if c in vowels)
        
        # Whitelist of valid tiny German words (2-3 chars)
        valid_tiny = {
            'der', 'die', 'das', 'dem', 'den', 'des', 'ein', 'eine', 'einer', 'eines',
            'und', 'mit', 'auf', 'für', 'von', 'aus', 'bei', 'vor', 'nach', 'über',
            'ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr', 'sie', 'uns', 'mir', 'dir',
            'ist', 'sind', 'war', 'ja', 'nein', 'doch', 'als', 'wie', 'so', 'nur',
            'bis', 'ab', 'zu', 'an', 'im', 'am', 'um', 'da'
        }
        
        if len(w) <= 3 and w not in valid_tiny:
            # If it's a 2-3 char word not in our whitelist, ensure it has at least one vowel
            # AND isn't just a consonant cluster.
            if v_count == 0:
                return True
            # Optional: reject 3-letter words with rare consonant patterns
            # (Keeping it simple for now)

        # Vowel density: Words should usually have at least one vowel per ~3.5 chars
        # Except for very long compounds which might have clusters.
        if len(w) > 3 and v_count == 0:
            return True

    else:
        # Non-German generic check
        vowels = set('aeiouy')
        if len(w) < 6 and not any(c in vowels for c in w):
            return True
            
    # 4. Repeating character threshold (kills aaa, xxx, etc.)

    for char in set(w):
        if w.count(char) > 3 and len(w) < 10:
            return True
            
    return False


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
        # Allow unlimited word lengths (no clamping)
        lengths.append(max(3, len(stripped)))
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
        super().__init__(params)
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
            cache_file = "models/arabic_code/arabic_pos_cache.json"
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

