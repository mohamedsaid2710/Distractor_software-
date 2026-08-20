"""Dictionary layer of the Maze pipeline: candidate pools and POS knowledge.

This module owns everything about WHICH words may become distractors:

- `wordfreq_dict` — base class: builds length-indexed word pools from the
  `wordfreq` library, retrieves candidates by length + Zipf frequency band
  (with tiered widening/fallbacks in `get_potential_distractors`), and keeps
  the per-language POS caches that guard grammar and casing.
- `wordfreq_English_zipf_dict` / `wordfreq_German_zipf_dict` /
  `wordfreq_Arabic_zipf_dict` — language subclasses wiring in spaCy, HanTa +
  Stanza (hybrid verification), and Farasa respectively.
- `get_thresholds_en/de/ar` — map target words to (length, frequency) windows.

HOW a candidate is finally scored/chosen (surprisal, thresholds, fallback
stages) lives in `sentence_set.py`, not here.
"""

import wordfreq
import os
import re
import math
import random
import logging
import json
import bisect
from collections import defaultdict
from HanTa import HanoverTagger as ht
from utils import strip_punct, ordered_unique, resolve_repo_path
from distractor import distractor_dict, distractor


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

def load_pos_lexicon(artifact_path, legacy_path, lang_label):
    """Load POS tags for *lang_label*, preferring the rebuildable artifact.

    Returns a plain {word: UPOS} dict.

    Two formats are accepted.  The artifact written by `build_lexicon.py` has a
    `_meta` stamp (tagger, version, build date) alongside its `tags`; the legacy
    cache is a bare {word: tag} mapping.  The artifact wins when present, and
    the legacy file remains the fallback so a machine with no taggers installed
    -- a cluster or Colab node -- still gets tags, which is what this load has
    always been for.  Deleting the artifact reverts to the legacy file.
    """
    # Same class of bug as the exclude lists: these are repo-relative paths and
    # only resolved while the CWD happened to be the repo root.
    artifact_path = resolve_repo_path(artifact_path)
    legacy_path = resolve_repo_path(legacy_path)

    if artifact_path and os.path.exists(artifact_path):
        try:
            with open(artifact_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("_meta", {})
            tags = data.get("tags", {})
            if tags:
                print(f"    [LEXICON] {lang_label}: {len(tags)} tags from "
                      f"{artifact_path} (tagger: {meta.get('tagger', 'unknown')}, "
                      f"built {meta.get('built', 'unknown')})", flush=True)
                return dict(tags)
            logging.warning("[LEXICON] %s has no 'tags' section; "
                            "falling back to %s", artifact_path, legacy_path)
        except Exception as e:
            logging.error("[LEXICON] Could not read %s (%s); falling back to %s",
                          artifact_path, e, legacy_path)

    if legacy_path and os.path.exists(legacy_path):
        try:
            with open(legacy_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"    [LEXICON] {lang_label}: {len(data)} tags from the legacy "
                  f"cache {legacy_path}. Rebuild with "
                  f"`python build_lexicon.py --lang {lang_label}` -- the legacy "
                  f"file was written append-only and cannot self-correct.",
                  flush=True)
            return dict(data)
        except Exception as e:
            logging.error("[LEXICON] Could not read %s: %s", legacy_path, e)

    logging.warning("[LEXICON] %s: no POS lexicon found; tagging happens at "
                    "run time if a tagger is installed.", lang_label)
    return {}



def _is_valid_cache_word(word: str, min_target_length: int = 3, lang: str = 'de', short_min_zipf: float = 3.5, mid_min_zipf: float = 3.0) -> bool:
    """
    Quality gate for words pulled from POS cache.
    Rejects garbage patterns that pass frequency checks but are corrupted (xte, odr, rav, etc).
    
    Args:
        word: Word to validate
        min_target_length: Minimum target word length (relaxes gates for short words)
        lang: Language code ('de' for German enforcement)
        short_min_zipf: Minimum Zipf for short words <= 5 chars
        mid_min_zipf: Minimum Zipf for medium words 6-8 chars
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
    
    # Gate 5: CRITICAL ZIPF VALIDATION
    # This used to run only when lang == 'de', so `short_min_zipf` (3.5 in
    # params_en.txt) and `min_zipf` were never enforced for English or Arabic
    # candidates -- which is how 'ocd', 'fyi', 'mla', 'yall', 'ive' and 'thy'
    # reached the output.  The thresholds stay length-conditional; only the
    # language restriction is gone.
    if word_len < 9:
        try:
            z = wordfreq.zipf_frequency(word_lower, lang)
            if word_len <= 5:
                if z < short_min_zipf:  # STRICT for short words
                    return False
            elif word_len <= 8:
                if z < mid_min_zipf:  # MEDIUM for medium words
                    return False
        except Exception:
            return False  # If wordfreq fails, reject the word
    
    return True


# Verified grammatical classes that may serve as NON-NOUN distractors.
# PROPN (proper-noun giveaway) and X (junk/unknown) are deliberately absent.
_SAFE_NON_NOUN_TAGS = ('VERB', 'ADJ', 'ADV', 'PRON', 'NUM', 'DET', 'PART',
                       'ADP', 'CCONJ', 'SCONJ', 'AUX', 'INTJ')

# German STTS tagset (used by HanTa) -> Universal POS. Shared by candidate
# tagging (`batch_tag_words`) and the Stanza/HanTa hybrid input verification
# (`batch_tag_inputs`). Anything not listed maps to 'X' (reject).
STTS_TO_UPOS = {
    'NN': 'NOUN', 'NE': 'PROPN',  # NE explicitly mapped to PROPN so the parameter controls it
    'ADJA': 'ADJ', 'ADJD': 'ADJ',
    'VVFIN': 'VERB', 'VVIMP': 'VERB', 'VVINF': 'VERB', 'VVIZU': 'VERB', 'VVPP': 'VERB',
    'VMFIN': 'VERB', 'VMINF': 'VERB', 'VMPP': 'VERB',
    'VAFIN': 'AUX', 'VAIMP': 'AUX', 'VAINF': 'AUX', 'VAPP': 'AUX',
    'ADV': 'ADV', 'PROAV': 'ADV', 'PTKA': 'ADV',
    'APPR': 'ADP', 'APPRART': 'ADP', 'APPO': 'ADP', 'APZR': 'ADP',
    'ART': 'DET', 'PDAT': 'DET', 'PIAT': 'DET', 'PIDAT': 'DET', 'PPOSAT': 'DET', 'PWAT': 'DET',
    'PDS': 'PRON', 'PIS': 'PRON', 'PPER': 'PRON', 'PPOSS': 'PRON', 'PRELS': 'PRON', 'PRF': 'PRON', 'PWS': 'PRON',
    'KON': 'CCONJ', 'KOUI': 'SCONJ', 'KOUS': 'SCONJ', 'KOKOM': 'SCONJ',
    'PTKZU': 'PART', 'PTKNEG': 'PART', 'PTKVZ': 'PART', 'PTKANT': 'PART',
    'ITJ': 'INTJ'
}


# Farasa morpheme classes that attach to a lexical head rather than being one.
# PRON belongs here: Arabic writes possessive and object pronouns as suffixes
# ("كتابه" = his book), so taking the last morpheme's tag labelled every such
# noun or verb a pronoun.  The shipped Arabic lexicon holds 7,166 PRON entries
# for that reason.
_FARASA_CLITICS = frozenset({
    'CONJ', 'PREP', 'DET', 'PART', 'PUNC', 'NSUFF', 'CASE', 'PRON', 'S', 'E',
})

# Farasa tag -> UPOS.  Anything not listed becomes 'X'.
_FARASA_TO_UPOS = {
    'NOUN': 'NOUN', 'V': 'VERB', 'ADJ': 'ADJ', 'ADV': 'ADV', 'NUM': 'NUM',
    'PREP': 'ADP', 'CONJ': 'CCONJ', 'DET': 'DET', 'PART': 'PART',
    'PRON': 'PRON', 'INTERJ': 'INTJ', 'ABBREV': 'X', 'FOREIGN': 'X',
}


def _farasa_morphemes(tagged):
    """Yield the (form, TAG) morphemes of a raw Farasa POS string, head-last.

    Current farasapy wraps its output in sentinels and separates tokens with
    spaces, so a one-word tagging looks like::

        'S/S كتاب/NOUN-MS E/E'
        'S/S مدرس +ة/NOUN+NSUFF-FS E/E'

    The previous parser split on '+' alone.  The whole string was therefore one
    "part", its tag read as the trailing 'E', and every Arabic word in the
    lexicon came back 'X' -- 'X' enters no candidate pool, so Arabic ran with no
    POS information at all and `exclude_propn_candidates` did nothing.
    Feature suffixes ('NOUN-MS', 'NSUFF-FS') are also stripped here; they made
    even a correctly-located tag fail the lookup.
    """
    for token in tagged.split():
        for morpheme in token.split('+'):
            morpheme = morpheme.strip()
            if '/' not in morpheme:
                continue
            form, _, tag = morpheme.rpartition('/')
            # 'NOUN-MS' -> 'NOUN'; the part after '-' is gender/number features.
            yield form, tag.strip().upper().split('-')[0]


def _farasa_tag_to_upos(tagged):
    """Convert a raw Farasa tag string to UPOS.

    Walks the morphemes from the END, skipping clitics to find the true lexical
    head.  If every morpheme is a clitic -- which is what a standalone pronoun
    or preposition looks like -- the last one is the head after all.
    """
    morphemes = [tag for _form, tag in _farasa_morphemes(tagged)
                 if tag not in ('S', 'E')]
    if not morphemes:
        return 'X'

    for tag in reversed(morphemes):
        if tag not in _FARASA_CLITICS:
            return _FARASA_TO_UPOS.get(tag, 'X')

    # All clitics: the word IS a function word, so take the last tag.
    return _FARASA_TO_UPOS.get(morphemes[-1], 'X')


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
        self.case_map = {}

    def get_words_by_len(self, desired_len):
        """Standard accessor for all words of a certain length."""
        pool = self.words_by_len.get(desired_len, [])
        # Return just the text for compatibility with old Stage 2 logic if still used
        return [w.text for w in pool]

    def get_best_frequency_pool(self, desired_len, target_zipf, n=400):
        """
        Retrieves a 'neighborhood' of words with frequencies closest to the target.
        Uses binary search for O(log N) neighborhood identification.

        *target_zipf* is a **Zipf** value (wordfreq's 1-7 scale), not the
        natural-log frequency stored on each ``distractor``.  The unit is in
        the parameter name deliberately: the two are a factor of ln(10) apart,
        and passing natlog here used to push the bisection key off the end of
        the array on every call, so the "neighborhood" returned was always the
        tail of the list — i.e. the rarest words of that length, identically
        for every target.
        """
        full_pool = self.words_by_len.get(desired_len, [])
        if not full_pool:
            return []
            
        if len(full_pool) <= n:
            return [w.text for w in full_pool]

        # `w.freq` is natural-log frequency (zipf * ln(10)); convert the key to
        # match rather than the whole array.
        target_freq = target_zipf * math.log(10)

        # Since full_pool is sorted by frequency DESCENDING, we bisect over the
        # negated frequencies, which are ascending.
        freqs_neg = [-w.freq for w in full_pool] # [-16.1, -15.9, ..., -2.3] (ascending)
        
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
            _sz = float(self.params.get('short_word_min_zipf', 3.5)) if hasattr(self, 'params') else 3.5
            _mz = float(self.params.get('min_zipf', 3.0)) if hasattr(self, 'params') else 3.0
            
            if not _is_valid_cache_word(lw, min_target_length=len(lw), lang=_lang, short_min_zipf=_sz, mid_min_zipf=_mz):
                continue
                
            l = len(lw)
            # TIGHTENED: A word is a noun if its category says so, OR if it's already in the noun pool
            # This prevents lowercased nouns from leaking into others_by_len
            if category == 'NOUN':
                self.nouns_by_len[l].add(lw)
            elif category in _SAFE_NON_NOUN_TAGS:
                # Double-check: if it has been marked as a noun elsewhere, don't put in 'others'
                if lw not in self.nouns_by_len[l]:
                    self.others_by_len[l].add(lw)
            # PROPN and X entries enter NEITHER pool: proper nouns are giveaways and
            # X marks junk/unknown — both must stay out of emergency fallback pools.

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

    def batch_tag_words(self, words, params=None, force_refresh=False):
        """Tag a list of words in bulk using SpaCy neural tagging. 
        
        Simple isolated word tagging (no context frames) to preserve pure two-mode logic.
        
        Args:
            words: List of words to tag
            params: Optional parameters dict
            force_refresh: If True, ignore cache and re-tag all words (default: False)
        """
        if self.nlp_sp is None or not words:
            if self.nlp_sp is None:
                print(f"[DIAG] batch_tag_words SKIPPED: nlp_sp is None", flush=True)
            return

        if not hasattr(self, 'pos_cache'):
            self.pos_cache = {}

        # ABSOLUTE TRUTH GUARD: Only tag words that are NOT already in our verified cache.
        # If force_refresh=True, ignore cache and tag all words.
        unique_words = ordered_unique(w.lower() for w in words if force_refresh or w.lower() not in self.pos_cache)
        
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
            display_count = False
            
            for i in range(0, len(unique_words), batch_size):
                chunk = unique_words[i:i + batch_size]
                
                if not display_count:
                    print(f"    [NLP] Running SpaCy tagging on {len(unique_words)} candidates...", flush=True)
                    display_count = True
                
                # English context-free tagging using SpaCy
                # spacy's pipe is highly optimized for batches
                docs = list(self.nlp_sp.pipe(chunk, disable=['parser', 'ner']))
                for word_l, doc in zip(chunk, docs):
                    if len(doc) > 0:
                        upos = doc[0].pos_
                    else:
                        upos = 'X'
                    self.pos_cache[word_l] = upos
                    self.case_map[word_l] = word_l.capitalize() if upos == 'PROPN' else None
                    
        except Exception as e:
            print(f"[DIAG] batch_tag_words SpaCy failed: {e}", flush=True)
            for word_l in unique_words:
                self.case_map[word_l] = None
                self.pos_cache[word_l] = 'X'



    def get_potential_distractors(self, min_length, max_length, min_freq, max_freq, params, pos_filter=None):
        """Returns list of candidates, using heuristic first, then widening, then batch SpaCy validation."""
        _lang = getattr(self, 'lang', 'de')
        n = int(params.get('num_to_test', 200))
        # How many candidates to aim for before the tiers give up widening.
        #
        # This used to be `max(n * 5, 2000)` under a pos_filter -- i.e. 2000 for
        # every English and German target.  A frequency band of +/- 1 Zipf at a
        # single exact length holds nothing like 2000 words, so the widening
        # loop below ran to its limit on EVERY target and turned the declared
        # band into +/- 4.9 Zipf.  Mode B then picks the maximum-surprisal
        # candidate, which sits at the rare edge, so every distractor landed far
        # below its target's frequency.  A smaller in-band pool beats a large
        # out-of-band one, because selection ranks within the pool either way.
        pool_factor = float(params.get('pool_size_factor', 2.0 if pos_filter else 1.0))
        target_pool_size = max(int(n * pool_factor), 50)
        
        # Get exclude list from params for pre-filtering
        exclude_words_set = set()
        exclude_path = params.get('exclude_words', None)
        if exclude_path:
            import os
            # NB: one `dirname` too many here used to resolve to the repo's
            # parent directory, silently dropping the exclude list.
            exclude_path = resolve_repo_path(exclude_path)
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
        
        # QUALITY GATE: reject web-scraping junk like 'fug', 'xte', 'uru' that
        # passes a frequency check.  This used to be wrapped in
        # `if max_length <= 5:`, so any target of 6+ characters got no
        # structural filtering at all -- the gate's own thresholds are already
        # length-conditional, so the outer guard only disabled it.
        _sz = float(params.get('short_word_min_zipf', 3.5)) if params else 3.5
        _mz = float(params.get('min_zipf', 3.0)) if params else 3.0

        def _quality_ok(w):
            return _is_valid_cache_word(strip_punct(w).lower(),
                                        min_target_length=min_length, lang=_lang,
                                        short_min_zipf=_sz, mid_min_zipf=_mz)

        distractor_opts = [w for w in distractor_opts if _quality_ok(w)]
        
        # PRE-FILTER: Remove excluded words
        if exclude_words_set:
            distractor_opts = [w for w in distractor_opts if strip_punct(w).lower() not in exclude_words_set]

        # TIER 1.25: Stepwise Frequency-Band Widening (`max_freq_widen`)
        # If the tight band starved, widen [min_freq, max_freq] symmetrically in
        # steps of `freq_widen_step` ZIPF units, up to `max_freq_widen` steps.
        # The step used to be 1 natural-log unit (0.43 Zipf) with 9 steps
        # allowed, so a fully-widened band was +/- 4.9 Zipf wider than declared
        # -- effectively unbounded.  Widening is now both smaller-grained and
        # reported, so a run says how far it had to go.
        max_freq_widen = int(params.get('max_freq_widen', 9))
        widen_step = float(params.get('freq_widen_step', 0.1)) * math.log(10)
        widen = 0
        if min_freq is not None and max_freq is not None:
            while len(distractor_opts) < target_pool_size and widen < max_freq_widen:
                widen += 1
                wider_pool = self.get_words(min_length, max_length,
                                            min_freq - widen * widen_step,
                                            max_freq + widen * widen_step,
                                            pos_filter=None, use_spacy=False)
                wider_pool = [w for w in wider_pool if _quality_ok(w)]
                if exclude_words_set:
                    wider_pool = [w for w in wider_pool if strip_punct(w).lower() not in exclude_words_set]
                if wider_pool:
                    distractor_opts = ordered_unique(list(distractor_opts) + list(wider_pool))
            if widen:
                logging.info(
                    "Frequency band widened by %.2f Zipf (%d step(s)) to reach %d "
                    "candidates of length %d-%d.",
                    widen * float(params.get('freq_widen_step', 0.1)), widen,
                    len(distractor_opts), min_length, max_length)
        self.last_freq_widen_steps = widen

        # TIER 1.5: Adaptive Frequency Widening
        # If tight band failed, use the optimized neighborhood search (binary search)
        if len(distractor_opts) < target_pool_size and 'target_zipf' in params:
            target_zipf = params['target_zipf']
            logging.info(f"Tight band search starved. Using neighborhood search around Zipf {target_zipf:.2f}")
            
            # Neighborhood size must stay a *neighborhood*: n=1000 against a
            # length bucket of ~3-4k words is a third of the lexicon, which
            # throws away the frequency match this tier exists to preserve.
            target_exact_len = params.get('target_exact_length') or ((min_length + max_length) // 2)
            neighbor_pool = self.get_best_frequency_pool(target_exact_len, target_zipf,
                                                         n=target_pool_size)
            
            # Filter and add to candidates
            if exclude_words_set:
                neighbor_pool = [w for w in neighbor_pool if w not in exclude_words_set]
            
            distractor_opts.extend(neighbor_pool)
            distractor_opts = ordered_unique(distractor_opts)

        # TIER 2: Quality Gate Fallback
        # If still starving, we can pull from other nearby lengths using the same frequency search
        if len(distractor_opts) < target_pool_size:
            target_exact_len = params.get('target_exact_length') or ((min_length + max_length) // 2)
            target_zipf = params.get('target_zipf', 5.0)
            _len_floor = int(params.get('min_word_len', DEFAULT_MIN_WORD_LEN))
            for length_diff in [1, -1]: # Try nearby length neighborhoods
                adj_len = target_exact_len + length_diff
                if adj_len >= _len_floor:
                    adj_pool = self.get_best_frequency_pool(adj_len, target_zipf,
                                                            n=target_pool_size)
                    if exclude_words_set:
                        adj_pool = [w for w in adj_pool if w not in exclude_words_set]
                    distractor_opts.extend(adj_pool)
                    distractor_opts = ordered_unique(distractor_opts)
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
                    _len_floor = int(params.get('min_word_len', DEFAULT_MIN_WORD_LEN))
                    for l in lengths_to_try:
                        if l < _len_floor: continue
                        # Use target_is_noun from params for faster pre-filtered expansion
                        target_is_noun_p = params.get('target_is_noun', False)
                        
                        if pos_filter == 'NOUN' or (pos_filter is None and target_is_noun_p):
                            pool = self.get_emergency_pool(l, is_noun=True)
                        elif pos_filter == '!NOUN' or (pos_filter is None and not target_is_noun_p):
                            pool = self.get_emergency_pool(l, is_noun=False)
                        else:
                            pool = self.get_emergency_pool(l, is_noun=True) + self.get_emergency_pool(l, is_noun=False)
                        
                        distractor_opts.extend(pool)
                        distractor_opts = ordered_unique(distractor_opts)
                    
                    if len(distractor_opts) >= target_pool_size:
                        break
                if len(distractor_opts) >= target_pool_size:
                    break

        # 3. --- HYPER-SPEED OPTIMIZATION: BATCH TAGGING ---
        if getattr(self, 'nlp_sp', None) is not None or getattr(self, 'hanta', None) is not None:
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
                is_noun = (tag == 'NOUN')
                
                # NEW STRICT RULE: If we specifically asked for !NOUN, ONLY allow verified grammatical tags.
                # If a word is tagged 'X' or is not in the safe non-noun list, REJECT IT.
                if pos_filter == '!NOUN':
                    if tag not in ('VERB', 'ADJ', 'ADV', 'PRON', 'NUM', 'DET', 'PART', 'ADP', 'CCONJ', 'SCONJ', 'AUX'):
                        continue
            
            # Generic fallback if not already determined in cache
            if not is_noun and not in_cache:
                if pos_filter == '!NOUN':
                    # DYNAMIC TAGGING FALLBACK
                    # If we need a !NOUN but don't know it, check it live using NLP models
                    safe_tag = None
                    try:
                        if _lang == 'de' and getattr(self, 'hanta', None) is not None:
                            self.batch_tag_words([w])
                            safe_tag = self.pos_cache.get(w_lower, 'X')
                        elif _lang == 'ar' and getattr(self, 'nlp_sp', None) is not None:
                            # NOTE: deliberately NOT `_farasa_tag_to_upos` — this
                            # variant keeps unknown Farasa tags as-is instead of
                            # collapsing them to 'X', so the safe-tag membership
                            # check below (not the parser) makes the reject call.
                            tagged = self.nlp_sp.tag(w)
                            parts = tagged.split('+')
                            pos = 'X'
                            for part in reversed(parts):
                                if '/' in part:
                                    curr = part.split('/')[-1].strip().upper()
                                    if curr not in ('CONJ', 'PREP', 'DET', 'PART', 'PUNC'):
                                        pos = curr
                                        break
                            if pos == 'X' and parts and '/' in parts[-1]:
                                pos = parts[-1].split('/')[-1].strip().upper()
                            farasa_map = {'V': 'VERB', 'ADJ': 'ADJ', 'PRON': 'PRON', 'NUM': 'NUM', 'ADV': 'ADV', 'PART': 'PART', 'NOUN': 'NOUN', 'PROPN': 'PROPN', 'PREP': 'ADP', 'CONJ': 'CCONJ'}
                            safe_tag = farasa_map.get(pos, pos)
                        elif _lang == 'en' and getattr(self, 'nlp_sp', None) is not None:
                            doc = self.nlp_sp(w)
                            if len(doc) > 0:
                                safe_tag = doc[0].pos_
                    except Exception:
                        pass
                        
                    if safe_tag and hasattr(self, 'pos_cache'):
                        self.pos_cache[w_lower] = safe_tag
                        
                    if not safe_tag or safe_tag not in ('VERB', 'ADJ', 'ADV', 'PRON', 'NUM', 'DET', 'PART', 'ADP', 'CCONJ', 'SCONJ', 'AUX'):
                        continue
                        
                    is_propn = (safe_tag == 'PROPN')
                    is_noun = (safe_tag == 'NOUN')
                else:
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
        
        # Initialize SpaCy for English NLP
        try:
            import spacy
            for model_name in ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm']:
                try:
                    self.nlp_sp = spacy.load(model_name)
                    break
                except Exception:
                    continue
            if self.nlp_sp is None:
                print("[WARN] SpaCy is installed, but no English models ('lg', 'md', or 'sm') were found. Tagging will skip. Run: python -m spacy download en_core_web_lg", flush=True)
        except ImportError:
            print("[WARN] SpaCy not found. English distractor tagging will skip. Run: pip install spacy", flush=True)

        exclude = params.get("exclude_words", "exclude_en.txt")
        include = params.get("include_words", None)
        lowercase_only = bool(params.get("lowercase_only", True))
        min_word_len = int(params.get("min_word_len", 3))
        min_zipf = float(params.get("min_zipf", 3.0))

        exclusions_lower = set()
        if exclude is not None:
            import os
            exclude = resolve_repo_path(exclude)
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
        self.pos_cache.update(load_pos_lexicon(
            "models/english_code/english_pos_lexicon.json",
            "models/english_code/english_pos_cache.json", "en"))

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
    # Last-resort heuristic for words absent from pos_cache/nouns_by_len.
    # Only suffixes that are near-perfectly nominal in the verified cache belong
    # here. Removed after empirical audit (July 2026): "in" (in, ein, bin, darin…),
    # "or" (vor, bevor, davor, empor…), "ist" (ist, meist, preist… — 57% noun only).
    NOUN_SUFFIXES = (
        "ung", "heit", "keit", "schaft", "tion", "sion",
        "tät", "ismus", "ment", "ik", "anz", "enz",
        "eur", "ling", "erich", "innen"
    )

    """Zipf-based German dictionary built from the wordfreq library."""

    def __init__(self, params={}):
        super().__init__(params)
        self.lang = "de"
        self.nlp_sp = None

        exclude = params.get("exclude_words", "exclude_de.txt")
        min_word_len = int(params.get("min_word_len", 3))
        min_zipf = float(params.get("min_zipf", 3.0))
        short_word_min_zipf = float(params.get("short_word_min_zipf", 3.5))

        exclusions_lower = set()
        if exclude is not None:
            exclude = resolve_repo_path(exclude)
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
        self.pos_cache.update(load_pos_lexicon(
            "models/german_code/german_pos_lexicon.json",
            "models/german_code/german_pos_cache_v2.json", "de"))
        # Seed nouns_by_len and others_by_len pools with the loaded tags
        self._populate_pools_from_cache()

        # HanTa Morphological Dictionary Bouncer
        self.hanta = None
        try:
            import HanTa.HanoverTagger as ht
            self.hanta = ht.HanoverTagger('morphmodel_ger.pgz')
            print("    [HanTa] Hannover Tagger loaded for strict dictionary lookups.", flush=True)
        except ImportError:
            print("    [HanTa] WARNING: HanTa not installed. Run: pip install HanTa", flush=True)
        except NameError:
            print("    [HanTa] WARNING: HanoverTagger (ht) not imported globally.", flush=True)
        except Exception as e:
            print(f"    [HanTa] WARNING: HanTa model not loaded ({e}).", flush=True)

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
            if upos == 'NOUN':
                self.case_map[word_l] = word_l.capitalize()
                self.nouns_by_len[l].add(word_l)
                if word_l in self.others_by_len[l]:
                    self.others_by_len[l].remove(word_l)
            else:
                self.case_map[word_l] = None
                if word_l in self.nouns_by_len[l]:
                    self.nouns_by_len[l].remove(word_l)
                # Only verified grammatical classes may act as non-noun distractors;
                # PROPN (giveaway) and X (junk) stay out of the emergency pools.
                if upos in _SAFE_NON_NOUN_TAGS:
                    self.others_by_len[l].add(word_l)
                elif word_l in self.others_by_len[l]:
                    self.others_by_len[l].remove(word_l)


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

    def _eval_single_word_case(self, token_lower):
        """POS check using HanTa Morphological Dictionary (Replacing Stanza)."""
        if not hasattr(self, 'pos_cache'):
            self.pos_cache = {}

        if getattr(self, 'hanta', None) is not None:
            # Re-use the flawless batch engine rather than writing duplicate HanTa logic
            self.batch_tag_words([token_lower])
        else:
            self.case_map[token_lower] = None

        return self.case_map.get(token_lower)

    def has_titlecase_variant(self, token):
        """Public check for noun/titlecase status of a token (lowercase or otherwise).
        
        Strategy: Use runtime-tagged POS cache (from batch_tag_words) which has correct Stanza tags.
        Falls back to titlecase heuristic if not yet tagged.
        Skips the pre-loaded 634k JSON cache (which may be corrupted).
        """
        t_lower = token.lower()
        
        # PRIORITY 0: Manual Overrides
        if hasattr(self, 'overrides') and t_lower in self.overrides:
            return self.overrides[t_lower] == 'NOUN'

        # PRIORITY 1: Check RUNTIME POS cache (populated by batch_tag_words with correct Stanza)
        if hasattr(self, 'pos_cache') and t_lower in self.pos_cache:
            pos_tag = self.pos_cache[t_lower]
            result = pos_tag == 'NOUN'
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

    def batch_tag_inputs(self, input_stanza_tags, params=None):
        """
        HYBRID APPROACH FOR INPUT WORDS:
        input_stanza_tags: dict of {word_lower: set_of_stanza_UPOS_tags}
        
        Reads the input words that were previously POS tagged by Stanza.
        Checks them against HanTa. 
        If Stanza and HanTa agree, they are cached as 100% correct.
        This safely grows the cache using verified context-aware data!
        Distractors remain purely tagged by HanTa.
        """
        if not self.hanta or not input_stanza_tags:
            return

        if not hasattr(self, 'pos_cache'):
            self.pos_cache = {}

        # Clean words and aggregate tags
        clean_stanza_tags = {}
        for w_lower, tags in input_stanza_tags.items():
            w_clean = re.sub(r"[^\w\säöüÄÖÜß]", "", w_lower).strip()
            if not w_clean: continue
            if w_clean not in clean_stanza_tags:
                clean_stanza_tags[w_clean] = set()
            clean_stanza_tags[w_clean].update(tags)

        to_tag = []
        for w_clean in clean_stanza_tags.keys():
            if w_clean not in self.pos_cache:
                to_tag.append(w_clean)
        
        to_tag = ordered_unique(to_tag)
        if not to_tag: return

        stts_map = STTS_TO_UPOS  # shared module-level STTS -> UPOS map

        print(f"    [Hybrid] Verifying {len(to_tag)} Stanza-tagged inputs against HanTa...", flush=True)
        
        new_entries = 0
        for w in to_tag:
            stanza_tags = clean_stanza_tags.get(w, set())
            if not stanza_tags or ('X' in stanza_tags and len(stanza_tags) == 1):
                continue 
                
            # Same rule as candidate tagging -- these two used to hold two
            # slightly different copies of it, so an input word and a candidate
            # word could get different tags from the same tagger.
            upos = self.hanta_upos(w)

            # The Verification Rule: If HanTa confirms any of the Stanza tags, we cache it
            if upos in stanza_tags and upos != 'X':
                self._record_pos_tag(w, upos)
                new_entries += 1
                
        if new_entries > 0:
            # In-memory only.  Persisting here made a run's own tagging decisions
            # an input to the next run (same input + same seed, different
            # output), and because the disk write was append-only a wrong tag
            # could never be corrected afterwards.  Rebuild the lexicon with
            # `python build_lexicon.py --lang de` instead.
            print(f"    [Hybrid] Verified {new_entries} new input words (in-memory).", flush=True)

    def batch_tag_words(self, words, params=None, force_refresh=False):
        """Batch-tag German candidates strictly using HanTa Morphological Dictionary."""
        if not self.hanta or not words:
            return

        if not hasattr(self, 'pos_cache'):
            self.pos_cache = {}

        to_tag = []
        for w in set(words):
            w_lower = w.lower()
            w_clean = re.sub(r"[^\w\säöüÄÖÜß]", "", w_lower).strip()
            if not w_clean: continue
            if force_refresh or w_clean not in self.pos_cache:
                to_tag.append(w_clean)
        
        to_tag = ordered_unique(to_tag)
        if not to_tag: return

        stts_map = STTS_TO_UPOS  # shared module-level STTS -> UPOS map

        print(f"    [HanTa] Looking up {len(to_tag)} words in strict morphological dictionary...", flush=True)
        
        for w in to_tag:
            self._record_pos_tag(w, self.hanta_upos(w))

    def hanta_upos(self, word):
        """UPOS tag for a lowercase German candidate, via HanTa.

        Capitalisation is the signal HanTa needs to recognise a proper noun --
        lowercased, "alonso" comes back ADV, "daphne" a verb, "kev" a number --
        so the word must be capitalised before tagging.  But capitalising a
        *verb* produces a grammatically real nominalised infinitive: "abarbeiten"
        -> NNI, "abartig" -> NNA.  Neither tag is in STTS_TO_UPOS, so both
        became 'X', and an X entry enters no candidate pool at all.

        The previous attempt to undo that re-analysed any NOUN in lowercase and
        took the lowercase tag if it looked verbal, which demoted 527 genuine
        nouns ("artefakte", "antwerpen") because HanTa mangles a lowercased
        noun's lemma.

        So: trust the capitalised analysis, and consult the lowercase one only
        for the two nominalisation tags that capitalising itself produced.
        Measured over the whole German lexicon, this cuts unusable 'X' entries
        from 3,064 to 180, grows the noun pool by 1,632, and correctly
        identifies 1,353 more proper nouns.
        """
        _, tag = self.hanta.analyze(word.capitalize())
        # HanTa writes VV(INF) / ADJ(A); standard STTS is VVINF / ADJA.
        clean_tag = tag.replace('(', '').replace(')', '')

        if clean_tag == 'NE':
            return 'PROPN'
        if clean_tag in ('NNI', 'NNA'):
            # Nominalised infinitive / adjective: an artefact of capitalising.
            _, tag_low = self.hanta.analyze(word.lower())
            return STTS_TO_UPOS.get(tag_low.replace('(', '').replace(')', ''), 'X')
        if clean_tag == 'NN':
            return 'NOUN'
        return STTS_TO_UPOS.get(clean_tag, 'X')

    def _record_pos_tag(self, word, upos):
        """Cache-aware POS tagging that populates length-indexed pools.
        TIGHTENED: Ensures nouns never enter the 'others' pool even if tagged with 
        non-noun POS when lowercased.
        """
        if not word: return
        self.pos_cache[word] = upos
        l = len(word)
        
        is_noun = upos == 'NOUN'
        
        if is_noun:
            self.case_map[word] = word.capitalize()
            self.nouns_by_len[l].add(word)
            # Remove from 'others' if it was mistakenly added before
            if word in self.others_by_len[l]:
                self.others_by_len[l].remove(word)
        else:
            # If not a noun, only add to 'others' if not already known to be a noun
            # AND it belongs to a verified grammatical class (no PROPN/X junk).
            if word not in self.nouns_by_len[l]:
                self.case_map[word] = None
                if upos in _SAFE_NON_NOUN_TAGS:
                    self.others_by_len[l].add(word)
                elif word in self.others_by_len[l]:
                    self.others_by_len[l].remove(word)

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




# Length bounds the candidate lexicons can actually serve.  `min_word_len` also
# governs which words are loaded into the lexicon at all, so a target shorter
# than it has an empty bucket and cannot be length-matched.
DEFAULT_MIN_WORD_LEN = 3
MAX_TARGET_LENGTH = 15


def effective_target_length(length, params, max_len=MAX_TARGET_LENGTH):
    """Clamp a target's true length into the range the lexicon can serve.

    The threshold functions clamped the length they searched with while
    `candidate_length_ok` in sentence_set.py compared against the *true*
    length, so with `len_tolerance: 0` a 1- or 2-letter English target rejected
    every candidate the pool could offer and fell through the whole cascade to
    the desperation fallback.  17% of non-initial tokens in English_sample.txt
    are 1-2 characters.  Both sides now clamp identically, via this function.

    Pass `max_len=None` to skip the upper clamp (German does not cap length).
    """
    floor = DEFAULT_MIN_WORD_LEN
    if params is not None:
        try:
            floor = int(params.get('min_word_len', DEFAULT_MIN_WORD_LEN))
        except (TypeError, ValueError):
            pass
    out = max(floor, int(length))
    if max_len is not None:
        out = min(out, int(max_len))
    return out


# Default frequency-band half-width, in Zipf units.  A distractor is drawn from
# targetZipf +/- this value.  1.0 is the value params_de.txt has always carried;
# it is now the default for every language because the alternative branch was
# degenerate (see _freq_band).
DEFAULT_FREQ_TOLERANCE = 1.0

# Bounds of the untightened band, in natural-log frequency units
# (Zipf 1.30 .. 4.78).  Only used when freq_tolerance is explicitly null.
_WIDE_BAND_LOW = 3.0
_WIDE_BAND_HIGH = 11.0


def _freq_band(freqs, params):
    """Return (min_freq, max_freq) in natural-log units for a set of targets.

    With a ``freq_tolerance`` (Zipf units, default 1.0) the band is centred on
    the mean target frequency.  ``freq_tolerance: null`` restores the old wide
    band.

    The old wide branch read::

        min_freq = min(min(freqs), 11)
        max_freq = max(max(freqs), 3)

    with the two bounds the wrong way round, so for any target between 3 and 11
    natlog -- i.e. Zipf <= 4.78, which is most content words -- it collapsed to
    min_freq == max_freq == the target's own frequency, a band of width zero.
    That starved the tight-band search on every English and Arabic run (only
    params_de.txt set freq_tolerance) and pushed selection into the fallback
    tiers, which is the origin of the frequency-mismatched output.
    """
    tol = DEFAULT_FREQ_TOLERANCE
    if params is not None and 'freq_tolerance' in params:
        tol = params['freq_tolerance']

    if tol is None:
        return (min(min(freqs), _WIDE_BAND_LOW),
                max(max(freqs), _WIDE_BAND_HIGH))

    tol_natlog = float(tol) * math.log(10)
    mean_freq = sum(freqs) / len(freqs)
    return mean_freq - tol_natlog, mean_freq + tol_natlog


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
        lengths.append(effective_target_length(len(stripped), params, max_len=None))
        freqs.append(get_frequency_de(stripped))
    min_length = min(lengths)
    max_length = max(lengths)

    min_freq, max_freq = _freq_band(freqs, params)

    return min_length, max_length, min_freq, max_freq


def get_thresholds_en(words, params=None):
    """English thresholds based on English frequency.

    Supports tight frequency band matching via ``freq_tolerance`` in *params*.
    """
    lengths = []
    freqs = []
    for word in words:
        stripped = strip_punct(word)
        lengths.append(effective_target_length(len(stripped), params))
        freqs.append(get_frequency_en(stripped))
    min_length = min(lengths)
    max_length = max(lengths)

    min_freq, max_freq = _freq_band(freqs, params)

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
            import logging
            logging.getLogger("farasapy_logger").setLevel(logging.ERROR)
            from farasa.pos import FarasaPOSTagger
            # interactive=True spawns a persistent Java process, making it much faster
            self.nlp_sp = FarasaPOSTagger(interactive=True)
        except Exception as e:
            print(f"[WARN] Farasa POSTagger not found or failed to load. Install 'farasapy' for Arabic POS tagging. Error: {e}")

        try:
            from farasa.segmenter import FarasaSegmenter
            self.segmenter = FarasaSegmenter(interactive=True)
        except Exception as e:
            print(f"[WARN] Farasa Segmenter not found. Arabic distractors may contain 'waow' prefixes. Error: {e}")

        exclude = params.get("exclude_words", "exclude_ar.txt")
        include = params.get("include_words", None)
        min_word_len = int(params.get("min_word_len", 2))
        min_zipf = float(params.get("min_zipf", 3.0))

        exclusions_lower = set()
        if exclude is not None:
            import os
            exclude = resolve_repo_path(exclude)
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
        self.pos_cache.update(load_pos_lexicon(
            "models/arabic_code/arabic_pos_lexicon.json",
            "models/arabic_code/arabic_pos_cache.json", "ar"))

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

    def batch_tag_inputs(self, input_stanza_tags, params=None):
        """
        HYBRID APPROACH FOR INPUT WORDS in Arabic (Stanza + Farasa):
        input_stanza_tags: dict of {word_lower: set_of_stanza_UPOS_tags}
        
        Reads the input words that were previously POS tagged by Stanza using their context.
        Checks them against Farasa. 
        If Stanza and Farasa agree, they are cached as 100% correct.
        This safely grows the cache using verified context-aware data!
        Distractors remain purely tagged by Farasa.
        """
        if getattr(self, 'nlp_sp', None) is None or not input_stanza_tags:
            return

        if not hasattr(self, 'pos_cache'):
            self.pos_cache = {}

        # Clean words and aggregate tags
        clean_stanza_tags = {}
        for w_lower, tags in input_stanza_tags.items():
            w_clean = strip_arabic_diacritics(w_lower)
            if not w_clean: continue
            if w_clean not in clean_stanza_tags:
                clean_stanza_tags[w_clean] = set()
            clean_stanza_tags[w_clean].update(tags)

        to_tag = []
        for w_clean in clean_stanza_tags.keys():
            if w_clean not in self.pos_cache:
                to_tag.append(w_clean)
        
        to_tag = ordered_unique(to_tag)
        if not to_tag: return

        print(f"[HYBRID Cache] Validating {len(to_tag)} Arabic inputs via Stanza-Farasa consensus...", flush=True)

        added = 0
        for w in to_tag:
            try:
                pos = _farasa_tag_to_upos(self.nlp_sp.tag(w))

                # Hybrid Agreement Check!
                stanza_tags = clean_stanza_tags[w]
                if pos in stanza_tags and pos != 'X':
                    self.pos_cache[w] = pos
                    added += 1
            except Exception:
                pass

        if added > 0:
            print(f"    [HYBRID] Added {added} high-confidence Arabic POS entries to runtime cache.", flush=True)
            try:
                cache_file = "models/arabic_code/arabic_pos_cache.json"
                existing = {}
                if os.path.exists(cache_file):
                    with open(cache_file, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                
                # Read-only: merge anything the artifact already knows that this
                # run has not tagged.  The write that used to be here made
                # generation nondeterministic across runs (see the German note
                # in batch_tag_inputs above).
                for k, v in existing.items():
                    self.pos_cache.setdefault(k, v)
            except Exception as e:
                print(f"[HYBRID] Cache read failed: {e}")

    def batch_tag_words(self, words, params=None, force_refresh=False):
        """Tag Arabic words using Farasa POSTagger."""
        if getattr(self, 'nlp_sp', None) is None or not words:
            return

        if not hasattr(self, 'pos_cache'):
            self.pos_cache = {}

        unique_words = ordered_unique(w for w in words if force_refresh or w not in self.pos_cache)
        if not unique_words:
            return

        print(f"    [NLP] Running Farasa POS Tagger on {len(unique_words)} Arabic candidates...", flush=True)
        # Farasa interactive doesn't have a reliable batch API method, so run loops. 
        # Interactive mode avoids re-loading the jar for each call so it's quite fast.
        for w in unique_words:
            try:
                # Raw Farasa output looks like "S-و/CONJ+ال/DET+قمر/NOUN"
                self.pos_cache[w] = _farasa_tag_to_upos(self.nlp_sp.tag(w))
            except Exception:
                self.pos_cache[w] = 'X'

        # No disk write.  This one was a full json.dump OVERWRITE of the whole
        # cache mid-run, X entries included -- so unlike the German path it did
        # not even honour the documented append-only contract, and a crashed run
        # could leave a truncated lexicon behind.


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
        lengths.append(effective_target_length(len(stripped), params))
        freqs.append(get_frequency_ar(stripped))
    min_length = min(lengths)
    max_length = max(lengths)

    min_freq, max_freq = _freq_band(freqs, params)

    return min_length, max_length, min_freq, max_freq

