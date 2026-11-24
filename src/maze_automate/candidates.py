"""Dictionary layer of the Maze pipeline: candidate pools and POS knowledge.

This module owns everything about WHICH words may become distractors:

- `wordfreq_dict` — base class: builds length-indexed word pools from the
  `wordfreq` library, retrieves candidates by length + Zipf frequency band
  (with tiered widening/fallbacks in `get_potential_distractors`), and keeps
  the per-language POS caches that guard grammar and casing.
- `wordfreq_English_zipf_dict` / `wordfreq_German_zipf_dict` /
  `wordfreq_Arabic_zipf_dict` — language subclasses.  Candidate POS comes
  from the norms-built lexicons; spaCy (en) and HanTa (de) are fallbacks
  when a lexicon is missing.  Arabic has no fallback.
- `get_thresholds_en/de/ar` — map target words to (length, frequency) windows.

HOW a candidate is finally scored/chosen (surprisal, thresholds, fallback
stages) lives in `selection.py`, not here.
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
# NOTE: HanTa, stanza and fasttext are imported lazily, where they are used,
# so a missing model fails only the step that needs it.
from maze_automate.utils import (strip_punct, ordered_unique, resolve_repo_path, bare_form,
                                  lexicon_path)


class distractor:
    """General class for distractor words"""

    def __init__(self, text, freq, pos=None):
        self.text = text
        self.len = len(text)
        self.freq = freq
        self.pos = pos


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

def load_pos_lexicon(artifact_path, lang_label):
    """{word: UPOS} from the lexicon built by `tools/build_lexicon.py`; {} if unusable.

    A missing or unreadable lexicon is reported before the run starts
    (main.confirm_lexicon); here it simply yields no tags.
    """
    artifact_path = resolve_repo_path(artifact_path)
    if not artifact_path or not os.path.exists(artifact_path):
        logging.warning("[LEXICON] %s: no POS lexicon at %s.", lang_label, artifact_path)
        return {}
    try:
        with open(artifact_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logging.error("[LEXICON] Could not read %s (%s).", artifact_path, e)
        return {}
    meta = data.get("_meta", {})
    tags = data.get("tags", {})
    if tags:
        print(f"    [LEXICON] {lang_label}: {len(tags)} tags from {artifact_path} "
              f"(source: {meta.get('tagger') or meta.get('source', 'unknown')}, "
              f"built {meta.get('built', 'unknown')})", flush=True)
    return dict(tags)


def lexicon_source(artifact_path):
    """Short description of a lexicon's source for the start-up banner."""
    try:
        with open(artifact_path, "r", encoding="utf-8") as f:
            meta = json.load(f).get("_meta", {})
    except Exception:
        return "no lexicon found"
    name = meta.get("source") or meta.get("tagger") or "unknown source"
    return name.split(" (")[0] + " lexicon"


def lexicon_status(artifact_path):
    """'norms', 'tagger-built', 'missing' or 'unreadable' for a POS lexicon file.

    Anything but 'norms' means candidates would be tagged word by word at run
    time, which main.run_stuff asks the user about instead of doing silently.
    """
    path = resolve_repo_path(artifact_path)
    if not path or not os.path.exists(path):
        return "missing"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return "unreadable"
    if not data.get("tags"):
        return "unreadable"
    return "norms" if data.get("_meta", {}).get("source_kind") == "norms" else "tagger-built"


def _load_clitic_words(artifact_path):
    """Arabic words the lexicon leaves out: those carrying a proclitic, and
    common web misspellings ("misspelled", e.g. فتاه for فتاة)."""
    path = resolve_repo_path(artifact_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("clitics", [])) | set(data.get("misspelled", {}))
    except Exception:
        return set()


def _load_spellings(artifact_path):
    """{wordfreq form: real spelling} from a lexicon's "spellings" key.

    German only: wordfreq writes ß as ss, and tools/build_lexicon.py records which
    words take a ß in today's spelling.
    """
    artifact_path = resolve_repo_path(artifact_path)
    if not artifact_path or not os.path.exists(artifact_path):
        return {}
    try:
        with open(artifact_path, "r", encoding="utf-8") as f:
            return dict(json.load(f).get("spellings", {}))
    except Exception:
        return {}


def load_lexicon_extras(artifact_path):
    """(shares, from_norms) for an artifact built by tools/build_lexicon.py.

    `shares` maps a word to the proportion of its uses its dominant POS
    accounts for (the ambiguity filter, `pos_min_share`).  `from_norms` is
    True when the tags came from lexical norms rather than a tagger run on
    bare words; generation then does NOT tag words the lexicon lacks, because
    that would put exactly those bare-word guesses back.
    """
    artifact_path = resolve_repo_path(artifact_path)
    if not artifact_path or not os.path.exists(artifact_path):
        return {}, False
    try:
        with open(artifact_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}, False
    from_norms = data.get("_meta", {}).get("source_kind") == "norms"
    return dict(data.get("shares", {})), from_norms



# Letters used by the vowel / consonant gates in _is_valid_cache_word, one
# entry per language so that a rule written for one never applies to another.
# Arabic has none: its script writes short vowels as optional diacritics, so
# "has a vowel letter" says nothing about whether an Arabic word is real.
_SCRIPT_LETTERS = {
    "en": {"vowels": r"[aeiouy]", "consonants": r"[bcdfghjklmnpqrstvwxz]"},
    "de": {"vowels": r"[aeiouyäöü]", "consonants": r"[bcdfghjklmnpqrstvwxzß]"},
    "ar": None,
}


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
    
    # Gate 2: must contain a vowel of its own language.  Each language has
    # its own letters (_SCRIPT_LETTERS); this used to be one German set for
    # every language, without "u" -- so every Arabic word failed, and so did
    # bus, run, und, Hund.
    letters = _SCRIPT_LETTERS.get(lang)
    if letters is not None and not re.search(letters["vowels"], word_lower):
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
    if letters is not None and word_len >= 4:
        if not re.search(letters["consonants"], word_lower):
            return False
    
    # Gate 5: CRITICAL ZIPF VALIDATION
    # This used to run only when lang == 'de', so `short_min_zipf` (3.5 in
    # config/params_en.txt) and `min_zipf` were never enforced for English or Arabic
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


class wordfreq_dict:
    """General class of dictionaries"""

    def __init__(self, params=None):
        params = {} if params is None else params
        self.params = params
        self.words = []
        self.words_by_len = {}
        self.nouns_by_len = defaultdict(set)
        self.others_by_len = defaultdict(set)
        self.lang = None
        self.nlp_sp = None
        self.case_map = {}
        self.pos_share = {}
        self.lexicon_from_norms = False

    def _load_lexicon(self, artifact_path, lang_label):
        """Load POS tags (and, for a norms-built lexicon, shares) into pos_cache."""
        if not hasattr(self, 'pos_cache'):
            self.pos_cache = {}
        self.pos_cache.update(load_pos_lexicon(artifact_path, lang_label))
        self.pos_share, self.lexicon_from_norms = load_lexicon_extras(artifact_path)
        if self.lexicon_from_norms:
            print(f"    [LEXICON] {lang_label}: tags are dominant POS from lexical "
                  f"norms; words without an entry are not tagged at run time.",
                  flush=True)

    def runtime_tagging_allowed(self):
        """False when the lexicon is norms-based: no bare-word guessing."""
        return not getattr(self, 'lexicon_from_norms', False)

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
        # The vocabulary after every filter (exclude list, clitics, misspellings,
        # frequency floor).  The emergency pools draw only from it.
        self.vocab = {w.text for w in self.words}
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
        """Vocabulary words of `length`, nouns or non-nouns by their lexicon tag.

        The noun / non-noun sets are filled from the whole POS lexicon, which
        also holds words the vocabulary deliberately leaves out (Arabic
        clitic forms, excluded words).  Only vocabulary words are returned, so
        this last-resort pool cannot bring those back.
        """
        pool = self.nouns_by_len if is_noun else self.others_by_len
        vocab = getattr(self, 'vocab', None)
        words = pool.get(length, ())
        # sorted: a set's order changes between Python processes, and the pool
        # is shuffled with the run's seed afterwards -- an unsorted pool made
        # "same input + same seed" give different output.
        return sorted(w for w in words if vocab is None or w in vocab)

    def get_words(self, length_low, length_high, freq_low, freq_high):
        """Vocabulary words within a length range and a frequency band."""
        matches = []
        for l in range(length_low, length_high + 1):
            for word in self.words_by_len.get(l, []):
                if freq_low is not None and freq_high is not None:
                    if not (freq_low <= word.freq <= freq_high):
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
        if not self.runtime_tagging_allowed():
            return
        if self.nlp_sp is None or not words:
            if self.nlp_sp is None:
                print("[DIAG] batch_tag_words SKIPPED: nlp_sp is None", flush=True)
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
                batch_size = int(params.get('nlp_batch_size', 256))
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
        # The size at which the retrieval tiers stop trying.  Every tier below
        # TIER 1 relaxes the frequency band, the length window, or both, so
        # they fire only when the pool is genuinely too small to choose from.
        # The rescue tiers (1.5, 2) also fetch only this many nearest-frequency
        # words: they used to fetch num_to_test x pool_size_factor (4,000 for
        # German), which is most of a length bucket, not a neighbourhood.
        min_pool_size = int(params.get('min_pool_size', 60))
        
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
        distractor_opts = self.get_words(min_length, max_length, min_freq, max_freq)
        
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
        # Widen only when the pool is too small to choose from -- not to reach
        # num_to_test.  Those are different questions: a bigger pool
        # cannot be had without leaving the frequency band, and selection ranks
        # within the pool either way, so chasing the larger number simply trades
        # the frequency match away.  German asks for num_to_test 2000, which no
        # +/-1 Zipf band at one length can supply, so it widened on 43 of 45
        # positions and only 10 landed inside the declared band.
        widen = 0
        if min_freq is not None and max_freq is not None:
            while len(distractor_opts) < min_pool_size and widen < max_freq_widen:
                widen += 1
                wider_pool = self.get_words(min_length, max_length,
                                            min_freq - widen * widen_step,
                                            max_freq + widen * widen_step)
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
        # The band actually searched, after widening (the report shows this).
        self.last_band = ((min_freq - widen * widen_step, max_freq + widen * widen_step)
                          if min_freq is not None and max_freq is not None else None)

        # TIER 1.5: Adaptive Frequency Widening
        # If tight band failed, use the optimized neighborhood search (binary search)
        if len(distractor_opts) < min_pool_size and 'target_zipf' in params:
            target_zipf = params['target_zipf']
            logging.info(f"Tight band search starved. Using neighborhood search around Zipf {target_zipf:.2f}")
            
            # Neighborhood size must stay a *neighborhood*: n=1000 against a
            # length bucket of ~3-4k words is a third of the lexicon, which
            # throws away the frequency match this tier exists to preserve.
            target_exact_len = params.get('target_exact_length') or ((min_length + max_length) // 2)
            neighbor_pool = self.get_best_frequency_pool(target_exact_len, target_zipf,
                                                         n=min_pool_size)
            
            # Filter and add to candidates
            if exclude_words_set:
                neighbor_pool = [w for w in neighbor_pool if w not in exclude_words_set]
            
            distractor_opts.extend(neighbor_pool)
            distractor_opts = ordered_unique(distractor_opts)

        # TIER 2: Quality Gate Fallback
        # If still starving, we can pull from other nearby lengths using the same frequency search
        if len(distractor_opts) < min_pool_size:
            target_exact_len = params.get('target_exact_length') or ((min_length + max_length) // 2)
            target_zipf = params.get('target_zipf', 5.0)
            _len_floor = int(params.get('min_word_len', DEFAULT_MIN_WORD_LEN))
            for length_diff in [1, -1]: # Try nearby length neighborhoods
                adj_len = target_exact_len + length_diff
                if adj_len >= _len_floor:
                    adj_pool = self.get_best_frequency_pool(adj_len, target_zipf,
                                                            n=min_pool_size)
                    if exclude_words_set:
                        adj_pool = [w for w in adj_pool if w not in exclude_words_set]
                    distractor_opts.extend(adj_pool)
                    distractor_opts = ordered_unique(distractor_opts)
                if len(distractor_opts) >= min_pool_size:
                    break

        # TIER 3: GRADUAL EMERGENCY EXPANSION (Ultimate Quality fallback)
        # If thresholds are NOT met, gradually expand length tolerance.
        if len(distractor_opts) < min_pool_size and hasattr(self, 'get_emergency_pool'):
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
                        # Noun / non-noun pools only when pos_match is on (the
                        # caller passes a pos_filter); otherwise both.  With no
                        # filter this used to fall back to target_is_noun, so
                        # "off" still split the pools.
                        if pos_filter == 'NOUN':
                            pool = self.get_emergency_pool(l, is_noun=True)
                        elif pos_filter == '!NOUN':
                            pool = self.get_emergency_pool(l, is_noun=False)
                        else:
                            pool = self.get_emergency_pool(l, is_noun=True) + self.get_emergency_pool(l, is_noun=False)
                        
                        distractor_opts.extend(pool)
                        distractor_opts = ordered_unique(distractor_opts)
                    
                    if len(distractor_opts) >= min_pool_size:
                        break
                if len(distractor_opts) >= min_pool_size:
                    break

        # 3. --- HYPER-SPEED OPTIMIZATION: BATCH TAGGING ---
        if getattr(self, 'nlp_sp', None) is not None or getattr(self, 'hanta', None) is not None:
            self.batch_tag_words(distractor_opts, params=params)
        
        # Pre-compute quality gate params once (German only)

        # 4. PRE-FILTER, before the pool is cut to `num_to_test`.  The same
        # rules for every language; each language's tags come from its own
        # lexicon.  sentence_set.admissibility() makes the final call on every
        # candidate -- this only keeps words that would certainly be rejected
        # there from taking places in the pool.
        #   pos_filter 'NOUN'  (pos_match on, noun target)     -> NOUN / PROPN
        #   pos_filter '!NOUN' (pos_match on, non-noun target) -> a verified
        #                                                         non-noun class
        #   no pos_filter      (pos_match 0)                   -> no class rule
        # This used to enforce the noun side for German only, so an English or
        # Arabic noun slot scored ~40% verbs and adjectives that were then
        # rejected, and chose from ~118 candidates instead of 200.
        exclude_propn = params.get('exclude_propn_candidates', False)
        from_norms = getattr(self, 'lexicon_from_norms', False)
        cache = getattr(self, 'pos_cache', {})

        filtered = []
        for w in distractor_opts:
            if not any(c.isalpha() for c in w):
                continue
            if '-' in w and all(len(part) < 2 for part in w.split('-')):  # x-x-x
                continue

            tag = cache.get(w.lower())
            if tag == 'X':
                continue                    # junk in every language
            if tag is None:
                if pos_filter:
                    continue                # class cannot be checked
                if exclude_propn and from_norms:
                    continue                # unlisted in norms: likely a name
            if exclude_propn and tag == 'PROPN':
                continue
            if pos_filter == 'NOUN' and tag not in ('NOUN', 'PROPN'):
                continue
            if pos_filter == '!NOUN' and tag not in _SAFE_NON_NOUN_TAGS:
                continue
            filtered.append(w)
        distractor_opts = filtered

        # 5. LIMIT SIZE (Return Target Pool)
        random.shuffle(distractor_opts)
        return distractor_opts[:n]


class wordfreq_English_zipf_dict(wordfreq_dict):
    LANG = "en"  # checked against the params file's `language` (pipeline.py)
    LEXICON_PATH = lexicon_path("en")
    FALLBACK_TAGGER = "spaCy"

    def _load_candidate_tagger(self):
        """spaCy for tagging bare candidate words (tagger-built lexicons only)."""
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
    """Zipf-based English dictionary with German-style filtering knobs.

    Supported params:
    - min_zipf (float, default 3.0)
    - min_word_len (int, default 3)
    - lowercase_only (bool, default True)
    - include_words (path, optional)
    - exclude_words (path, optional)
    """

    def __init__(self, params=None):
        params = {} if params is None else params
        super().__init__(params)
        self.lang = "en"
        self.nlp_sp = None
        if load_lexicon_extras(self.LEXICON_PATH)[1]:
            # Norms-built lexicon: candidate POS is a lookup, so the bare-word
            # tagger is loaded only as a fallback when no lexicon is present.
            print("    [LEXICON] en: norms lexicon -- candidate tagger not loaded.", flush=True)
        else:
            self._load_candidate_tagger()

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
        self._load_lexicon(self.LEXICON_PATH, "en")

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
    LANG = "de"  # checked against the params file's `language` (pipeline.py)
    LEXICON_PATH = lexicon_path("de")
    FALLBACK_TAGGER = "HanTa"
    # Apply the lexicon's ß spellings to wordfreq's ss-forms.  tools/build_lexicon.py
    # turns this off so every build starts from wordfreq's own forms.
    USE_SPELLINGS = True

    def _load_candidate_tagger(self):
        """HanTa for tagging bare candidate words (tagger-built lexicons only)."""
        try:
            from HanTa import HanoverTagger as ht
            self.hanta = ht.HanoverTagger('morphmodel_ger.pgz')
            print("    [HanTa] Hannover Tagger loaded for strict dictionary lookups.", flush=True)
        except ImportError:
            print("    [HanTa] WARNING: HanTa not installed. "
                  "Run `uv sync` to install all dependencies.", flush=True)
        except Exception as e:
            print(f"    [HanTa] WARNING: HanTa model not loaded ({e}).", flush=True)
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

    def __init__(self, params=None):
        params = {} if params is None else params
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
        if not hasattr(self, 'common_casing'):
            self.common_casing = {}

        # === PRELOAD GERMAN POS CACHE ===
        self._load_lexicon(self.LEXICON_PATH, "de")
        # Seed nouns_by_len and others_by_len pools with the loaded tags
        self._populate_pools_from_cache()

        self.hanta = None
        if self.lexicon_from_norms:
            print("    [LEXICON] de: norms lexicon -- candidate tagger (HanTa) not loaded.", flush=True)
        else:
            self._load_candidate_tagger()

        freq_dict = wordfreq.get_frequency_dict("de")
        source_words = list(freq_dict.keys())
        # wordfreq writes ß as ss; the lexicon says which words take a ß.
        spellings = _load_spellings(self.LEXICON_PATH) if self.USE_SPELLINGS else {}

        self.words = []
        seen = set()
        for raw in source_words:
            lw = raw.strip().lower()
            lw = spellings.get(lw, lw)
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
        if not self.runtime_tagging_allowed():
            return
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
            # `python tools/build_lexicon.py --lang de` instead.
            print(f"    [Hybrid] Verified {new_entries} new input words (in-memory).", flush=True)

    def batch_tag_words(self, words, params=None, force_refresh=False):
        """Batch-tag German candidates strictly using HanTa Morphological Dictionary."""
        if not self.runtime_tagging_allowed():
            return
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
    `candidate_length_ok` in selection.py compared against the *true*
    length, so with `len_tolerance: 0` a 1- or 2-letter English target rejected
    every candidate the pool could offer and fell through the whole cascade to
    the desperation fallback.  17% of non-initial tokens in examples/english_sample.txt
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
# targetZipf +/- this value.  1.0 is the value config/params_de.txt has always carried;
# it is now the default for every language because the alternative branch was
# degenerate (see _freq_band).
DEFAULT_FREQ_TOLERANCE = 1.0

# Bounds of the untightened band, in natural-log frequency units
# (Zipf 1.30 .. 4.78).  Only used when freq_tolerance is explicitly null.
_WIDE_BAND_LOW = 3.0
_WIDE_BAND_HIGH = 11.0


# Short function words: no word but another function word is as frequent as
# "der" or "the", so a band around the target only offers those, and they are
# too predictable to pass the surprisal threshold.  With
# `short_word_freq_match: False` such a target's band is
# [short_word_band_floor, _SHORT_BAND_TOP] (Zipf) instead: length is still
# matched, frequency only has a floor.  Short content words (Tee, dog) keep
# their band.
DEFAULT_SHORT_BAND_FLOOR = 4.0
_SHORT_BAND_TOP = 9.0      # above every word's frequency


def _is_short_target(lengths, params):
    """Short function-word target whose frequency band is dropped."""
    if not lengths or params is None:
        return False
    if params.get('short_word_freq_match', False):
        return False
    if not params.get('target_is_function_word', False):
        return False
    return max(lengths) <= int(params.get('short_word_max_len', 3))


def _freq_band(freqs, params, lengths=None):
    """Return (min_freq, max_freq) in natural-log units for a set of targets.

    A short function-word target (see _is_short_target) gets no band around
    its own frequency unless ``short_word_freq_match`` is True: only the
    floor ``short_word_band_floor`` applies.

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
    config/params_de.txt set freq_tolerance) and pushed selection into the fallback
    tiers, which is the origin of the frequency-mismatched output.
    """
    if _is_short_target(lengths, params):
        floor = float(params.get('short_word_band_floor', DEFAULT_SHORT_BAND_FLOOR))
        return floor * math.log(10), _SHORT_BAND_TOP * math.log(10)

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
        stripped = bare_form(word)
        lengths.append(effective_target_length(len(stripped), params, max_len=None))
        freqs.append(get_frequency_de(stripped))
    min_length = min(lengths)
    max_length = max(lengths)

    min_freq, max_freq = _freq_band(freqs, params, lengths)

    return min_length, max_length, min_freq, max_freq


def get_thresholds_en(words, params=None):
    """English thresholds based on English frequency.

    Supports tight frequency band matching via ``freq_tolerance`` in *params*.
    """
    lengths = []
    freqs = []
    for word in words:
        stripped = bare_form(word)
        lengths.append(effective_target_length(len(stripped), params))
        freqs.append(get_frequency_en(stripped))
    min_length = min(lengths)
    max_length = max(lengths)

    min_freq, max_freq = _freq_band(freqs, params, lengths)

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
    LANG = "ar"  # checked against the params file's `language` (pipeline.py)
    # Leave out the lexicon's clitic and misspelled words.  tools/build_lexicon.py
    # turns this off so every build starts from the full wordfreq list.
    USE_LEXICON_EXCLUSIONS = True
    LEXICON_PATH = lexicon_path("ar")
    # No bare-word tagger for Arabic.  Farasa was dropped: farasapy's automatic
    # download of its Java binaries is broken (Nov 2026), and it needed Java.
    # The CAMeL-built lexicon is required.
    FALLBACK_TAGGER = None

    """Zipf-based Arabic dictionary built from the wordfreq library.

    Arabic has no uppercase/lowercase distinction, so casing logic is skipped.
    Diacritics (tashkeel) are stripped for consistent matching.
    """

    def __init__(self, params=None):
        params = {} if params is None else params
        super().__init__(params)
        self.lang = "ar"
        self.nlp_sp = None

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
        self._load_lexicon(self.LEXICON_PATH, "ar")

        include_words = None
        if include is not None and os.path.exists(include):
            try:
                with open(include, "r", encoding="utf-8") as f:
                    include_words = [line.strip() for line in f if line.strip()]
            except Exception:
                include_words = None

        freq_dict = wordfreq.get_frequency_dict("ar")
        source_words = include_words if include_words is not None else freq_dict.keys()

        # Words with an attached w+/f+/b+/k+/l+ (e.g. والكتاب, بالبيت) are
        # obvious as distractors.  The lexicon lists them, decided once by
        # CAMeL Tools at build time (tools/build_lexicon.py).
        clitic_words = (_load_clitic_words(self.LEXICON_PATH)
                        if self.USE_LEXICON_EXCLUSIONS else set())

        self.words = []
        seen = set()

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
            
            if token in clitic_words:
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

    def get_titlecase_variant(self, token):
        """Arabic has no casing; always returns None."""
        return None

    def has_titlecase_variant(self, token):
        """Arabic has no title casing; always returns False."""
        return False

    def batch_tag_inputs(self, input_stanza_tags, params=None):
        """No-op: Arabic candidate POS comes only from the lexicon."""
        return

    def batch_tag_words(self, words, params=None, force_refresh=False):
        """No-op: Arabic candidate POS comes only from the lexicon."""
        return


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
        stripped = bare_form(word)
        lengths.append(effective_target_length(len(stripped), params))
        freqs.append(get_frequency_ar(stripped))
    min_length = min(lengths)
    max_length = max(lengths)

    min_freq, max_freq = _freq_band(freqs, params, lengths)

    return min_length, max_length, min_freq, max_freq

