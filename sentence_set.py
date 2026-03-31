import logging
from utils import copy_punct, strip_punct
from limit_repeats import Repeatcounter
import re
import random
import os

# Common uppercase acronyms to preserve when used as distractors in English.
_EN_ACRONYM_WHITELIST = {
    'ai', 'api', 'bbc', 'cia', 'cpu', 'eu', 'fbi', 'gdp', 'gps', 'gpu',
    'ibm', 'imf', 'ml', 'nasa', 'nato', 'oecd', 'uk', 'un', 'usa', 'uefa',
    'who',
}

_X_PLACEHOLDER_RE = re.compile(r"^x(?:-x)*$", re.IGNORECASE)

# lazy SpaCy pipeline (initialized on first use)
_spacy_nlp = {}

def _get_spacy_nlp(lang='de'):
    """Return a loaded spaCy pipeline for `lang` ('de' or 'en'), else None."""
    global _spacy_nlp
    key = 'en' if str(lang or '').lower().startswith('en') else 'de'
    if key in _spacy_nlp:
        return _spacy_nlp[key]
    try:
        import spacy
    except Exception:
        _spacy_nlp[key] = None
        return None

    model_names = ['en_core_web_sm', 'en_core_web_md'] if key == 'en' else ['de_core_news_sm', 'de_core_news_md']
    for model_name in model_names:
        try:
            _spacy_nlp[key] = spacy.load(model_name)
            return _spacy_nlp[key]
        except Exception:
            continue
    _spacy_nlp[key] = None
    return None


def _split_punct(token):
    m = re.match(r"^(?P<prefix>[^\wÄÖÜäöüß]*)(?P<body>[\wÄÖÜäöüß'-]+)(?P<suffix>[^\wÄÖÜäöüß]*)$", token, re.UNICODE)
    if not m:
        return '', token, ''
    return m.group('prefix'), m.group('body'), m.group('suffix')


# Global Cache for proper noun checks to speed up is_propn_candidate
PROPN_CACHE = {}

def is_propn_candidate(cand):
    """Returns True if the candidate is likely a Proper Noun (PROPN).
    Uses a cache to avoid redundant SpaCy calls.
    """
    clean_cand = strip_punct(cand)
    if not clean_cand:
        return False
        
    if clean_cand in PROPN_CACHE:
        return PROPN_CACHE[clean_cand]
        
    try:
        lang = 'de' # Default to German for now given context
        nlp_sp = _get_spacy_nlp(lang)
        if nlp_sp is not None:
            # We capitalize to test if it behaves as a noun/proper noun
            doc2 = nlp_sp(clean_cand.capitalize())
            is_propn = len(doc2) > 0 and doc2[0].pos_ == 'PROPN'
            PROPN_CACHE[clean_cand] = is_propn
            return is_propn
    except Exception:
        pass
    
    # Heuristic fallback if SpaCy fails
    is_propn = clean_cand[0].isupper() and len(clean_cand) > 1
    PROPN_CACHE[clean_cand] = is_propn
    return is_propn


def _is_x_placeholder_token(token):
    """True for placeholder tokens like x, x-x, x-x-x, ... (ignoring edge punct)."""
    if not token:
        return False
    body = strip_punct(token)
    return bool(_X_PLACEHOLDER_RE.fullmatch(body))


def _placeholder_for_length(length):
    """Build a first-position placeholder with one 'x' per character in the target word."""
    n = max(1, int(length))
    if n == 1:
        return "x"
    return "-".join(["x"] * n)


def _copy_edge_punct_no_case(source_token, token):
    """Copy only leading/trailing punctuation from source token, leave token case untouched."""
    prefix, _body, suffix = _split_punct(source_token)
    return prefix + token + suffix


def _looks_acronym(body):
    if not body or not body.isalpha():
        return False
    return body.isupper() and 2 <= len(body) <= 8


def _looks_titlecase_name(body):
    if not body or len(body) < 2 or not body.isalpha():
        return False
    return body[0].isupper() and body[1:].islower()


def _normalize_english_distractor_case(source_token, distractor_token):
    """English-specific casing normalization using source token as signal."""
    if _is_x_placeholder_token(distractor_token):
        return distractor_token
    _, src_body, _ = _split_punct(source_token)
    d_prefix, d_body, d_suffix = _split_punct(distractor_token)
    if not d_body:
        return distractor_token

    # If the replaced token is an acronym, force distractor to all-caps.
    if _looks_acronym(src_body):
        return d_prefix + d_body.upper() + d_suffix

    # Also capitalize known acronym distractors.
    if d_body.lower() in _EN_ACRONYM_WHITELIST:
        return d_prefix + d_body.upper() + d_suffix

    # If the source token looks like a proper name, keep Title Case.
    if _looks_titlecase_name(src_body):
        return d_prefix + d_body[:1].upper() + d_body[1:].lower() + d_suffix

    return distractor_token


def _detect_language(params):
    """Infer language code ('en', 'de', or 'ar') from params."""
    lang = (params.get('language', '') or '').strip().lower()
    if lang.startswith('de'):
        return 'de'
    if lang.startswith('ar'):
        return 'ar'
    if lang.startswith('en'):
        return 'en'
    model_loc = (params.get('model_loc', '') or '').lower()
    dict_cls = (params.get('dictionary_class', '') or '').lower()
    hf_name = (params.get('hf_model_name', '') or '').lower()
    if ('german' in model_loc) or ('german' in dict_cls) or ('dbmdz' in hf_name):
        return 'de'
    if ('arabic' in model_loc) or ('arabic' in dict_cls) or ('aragpt' in hf_name) or ('aubmindlab' in hf_name):
        return 'ar'
    return 'en'


def _get_german_grammatical_case(token, dict_obj):
    """Determine correct German casing based on the token's own linguistic class."""
    if _is_x_placeholder_token(token):
        return token
    prefix, body, suffix = _split_punct(token)
    if not body:
        return token

    # Check if it's a noun/proper noun in the dictionary
    title_var = None
    try:
        if hasattr(dict_obj, 'get_titlecase_variant'):
            title_var = dict_obj.get_titlecase_variant(body)
    except Exception:
        pass

    if title_var:
        new_body = title_var
    else:
        # Not a noun -> lowercase (Standard German rule)
        new_body = body.lower()
    
    return prefix + new_body + suffix

def _normalize_distractor_token(token, dict_obj, lang='de', source_pos=None):
    """Normalize casing for a single distractor token."""
    if lang == 'de':
        return _get_german_grammatical_case(token, dict_obj)
    
    # Non-German fallback (e.g. English as before)
    if _is_x_placeholder_token(token):
        return token
    prefix, body, suffix = _split_punct(token)
    if not body:
        return token
    return prefix + body.lower() + suffix

def no_duplicates(my_list):
    """True if list has no duplicates, else false"""
    return len(my_list) == len(set(my_list))


class Sentence:
    """a sentence to get distractors for
    has a list of words (in the sentence)
    a list of labels for matching with other sentences
    an id = item number
    and tag, which we do nothing with, but store anyhow
    """

    def __init__(self, words, labels, id, tag, original_sentence):
        if no_duplicates(labels):
            self.words = words  # list of words in sentence
            self.word_sentence = original_sentence  # sentence itself with punctuation
            self.labels = labels  # list of labels in sentence
            self.label_sentence = " ".join([str(lab) for lab in self.labels])  # labels as sentence
            self.id = id  # item number
            self.tag = tag  # group/type
            self.distractors = []
            self.distractor_sentence = ""
            self.probs = {}  # using a dictionary so we can start at 1 and not 0
            self.surprisal = {}
            self.hiddens = {}
        else:
            logging.error("duplicate labels on sentence %s", " ".join(words))
            raise ValueError("duplicate labels")

    def do_model(self, model):
        """Run the model and record surprisals at each position"""
        hidden = model.empty_sentence()  # initialize
        for i in range(len(self.words) - 1):  # use labels to index
            hidden, self.probs[self.labels[i + 1]] = model.update(hidden, self.words[i])
            # store a shallow copy of the hidden/context after adding this word
            try:
                self.hiddens[self.labels[i + 1]] = list(hidden)
            except Exception:
                # if hidden is not list-like, store as-is
                self.hiddens[self.labels[i + 1]] = hidden

    def do_surprisal(self, model):
        """Get surprisals of words in sentence"""
        for i in range(1, len(self.labels)):  # zeroeth position doesn't count
            lab = self.labels[i]
            # prefer model that can compute surprisal from hidden if available
            if hasattr(model, 'get_surprisal_from_hidden') and lab in self.hiddens:
                self.surprisal[lab] = model.get_surprisal_from_hidden(self.hiddens[lab], self.words[i])
            else:
                self.surprisal[lab] = model.get_surprisal(self.probs[lab], self.words[i])


class Label:
    """A set of words etc, associated with a label, within a sentence set"""

    def __init__(self, id, lab):
        self.id = id  # item number
        self.lab = lab
        self.words = []
        self.probs = []
        self.surprisals = []
        self.hiddens = []
        self.pos = []
        self.surprisal_targets = []

    def add_sentence(self, word, probs, surprisal, hidden=None):
        """Given a position that belongs in the label, add it's attributes to our lists"""
        self.words.append(word)
        self.probs.append(probs)
        self.surprisals.append(surprisal)
        self.hiddens.append(hidden)
        # pos will be filled by make_labels if available; placeholder None
        self.pos.append(None)

    def choose_distractor(self, model, dictionary, threshold_func, params, banned):
        """Given a parameters specified in params and stuff
        Find a distractor not on banned (banned=already used in same sentence set)
        That hopefully meets threshold"""
        lang = _detect_language(params)
        # Identify if we should strictly match noun POS (default False, except if enabled in params)
        match_noun_pos = bool(params.get('match_noun_pos', False))
        absolute_threshold_only = bool(params.get('absolute_threshold_only', False))

        # --- Early position boost ---
        # Raises the surprisal threshold for positions near the start of the
        # sentence where the language model has little context, producing more
        # clearly-implausible distractors and reducing participant error rates
        # at those positions.
        early_boost = float(params.get('early_position_boost', 0))
        early_count = int(params.get('early_position_count', 2))
        is_early = False
        try:
            label_idx = int(self.lab)
            # label 0 is the placeholder (first_token_placeholder), so
            # early positions are labels 1 through early_count.
            if 1 <= label_idx <= early_count:
                is_early = True
        except (ValueError, TypeError):
            pass

        for surprisal in self.surprisals:  # calculate desired surprisal thresholds
            if absolute_threshold_only:
                base = params["min_abs"]
            else:
                base = max(params["min_abs"], surprisal + params["min_delta"])
            if is_early:
                base += early_boost
            self.surprisal_targets.append(base)

        # decide whether this label refers to nouns (use POS info if available)
        noun_tags = set(['NOUN', 'PROPN'])
        target_is_noun = False
        try:
            # if any POS tag for this label is a noun/proper noun, consider it a noun label
            for p in self.pos:
                if p in noun_tags:
                    target_is_noun = True
                    break
        except Exception:
            target_is_noun = False
        # If POS info is unavailable or non-conclusive, fall back to dictionary hints
        if not target_is_noun:
            try:
                currency_whitelist = set(['euro', 'dollar', 'pound', 'yen'])
                for w in self.words:
                    wlow = w.lower()
                    try:
                        # if dict provides an explicit Titlecase variant, treat as noun
                        if hasattr(dictionary, 'has_titlecase_variant') and dictionary.has_titlecase_variant(w):
                            target_is_noun = True
                            break
                    except Exception:
                        pass
                    if wlow in currency_whitelist:
                        target_is_noun = True
                        break
            except Exception:
                pass

        exclude_propn_candidates = bool(params.get('exclude_propn_candidates', False))
        nlp_sp = _get_spacy_nlp(lang)

        _propn_cache = {}
        _pos_cache = {}
        
        def get_candidate_pos(candidate):
            if nlp_sp is None: return None
            clean = strip_punct(candidate)
            if not clean: return None
            key = clean.lower()
            if key in _pos_cache: return _pos_cache[key]
            try:
                doc = nlp_sp(clean)
                val = doc[0].pos_ if doc and len(doc) > 0 else None
            except Exception:
                val = None
            _pos_cache[key] = val
            return val

        target_pos = get_candidate_pos(self.words[0]) if self.words else None
        target_is_noun = (target_pos == 'NOUN')

        # --- CASING CALCULATION ---
        sw_target = strip_punct(self.words[0]) if self.words else ""
        target_is_lower = sw_target[0].islower() if sw_target else True

        # get us some distractor candidates
        # Strict POS matching: if target is a noun, dist must be a noun.
        # If target is NOT a noun, dist must NOT be a noun.
        pos_filter = None
        if match_noun_pos:
            pos_filter = 'NOUN' if target_is_noun else '!NOUN'
        
        min_length, max_length, min_freq, max_freq = threshold_func(self.words, params)
        distractor_opts = dictionary.get_potential_distractors(min_length, max_length, min_freq, max_freq, params, pos_filter=pos_filter)

        # --- ADAPTIVE LENGTH SEARCH (Fix for 0-candidate long words) ---
        # If no candidates found for exact length, widen search recursively.
        if not distractor_opts:
            for extra_len in range(1, 11): # Try widening up to +/- 10 characters
                logging.info(f"0 candidates for {self.words} at exact length. Widening search by +/- {extra_len}")
                distractor_opts = dictionary.get_potential_distractors(
                    min_length - extra_len, max_length + extra_len, 
                    min_freq, max_freq, params, pos_filter=pos_filter
                )
                if distractor_opts:
                    break
                
                # --- FINAL FALLBACK: Frequency Relaxation ---
                # If still nothing, try ignoring frequency thresholds for this length widening
                logging.info(f"Still 0 candidates. Relaxing frequency constraints for +/- {extra_len}")
                distractor_opts = dictionary.get_potential_distractors(
                    min_length - extra_len, max_length + extra_len, 
                    None, None, params, pos_filter=pos_filter
                )
                if distractor_opts:
                    break

        # --- Fallback: allow other non-noun classes as a last resort ---
        if match_noun_pos and not target_is_noun:
            num_req = int(params.get('num_to_test', 100))
            if len(distractor_opts) < (num_req // 2):
                logging.info(f"Not enough non-NOUN candidates for {self.words}, falling back to broader search (still excluding nouns)")
                fallback_opts = dictionary.get_potential_distractors(min_length, max_length, min_freq, max_freq, params, pos_filter="!NOUN")
                existing = set(distractor_opts)
                for opt in fallback_opts:
                    if opt not in existing:
                        distractor_opts.append(opt)

        if match_noun_pos and target_pos:
            exact = []
            others = []
            bad_pos = {'PROPN', 'X', 'SYM', 'INTJ'}
            if not target_is_noun:
                bad_pos.add('NOUN')
                
            # --- HYPER-SPEED CPU FIX: BATCH TAGGING ---
            try:
                if nlp_sp is not None and distractor_opts:
                    clean_opts = [strip_punct(c) for c in distractor_opts]
                    docs = list(nlp_sp.pipe(clean_opts, batch_size=500))
                    for c, doc in zip(distractor_opts, docs):
                        c_pos = doc[0].pos_ if doc and len(doc) > 0 else None
                        is_exact_match = (c_pos == target_pos) or (target_is_noun and c_pos == 'PROPN')
                        
                        if is_exact_match:
                            exact.append(c)
                        elif c_pos not in bad_pos:
                            others.append(c)
                else:
                    for c in distractor_opts:
                        c_pos = get_candidate_pos(c)
                        is_exact_match = (c_pos == target_pos) or (target_is_noun and c_pos == 'PROPN')
                        
                        if is_exact_match:
                            exact.append(c)
                        elif c_pos not in bad_pos:
                            others.append(c)
            except Exception as e:
                logging.error(f"Batch POS tagging failed: {e}")
                for c in distractor_opts:
                    c_pos = get_candidate_pos(c)
                    if c_pos == target_pos:
                        exact.append(c)
                    elif c_pos not in bad_pos:
                        others.append(c)

            # Pre-sort exact matches first, fallback second. Drop bad POS completely!
            if target_is_noun:
                distractor_opts = exact
            else:
                distractor_opts = exact + others

        enforce_length_match = bool(params.get('enforce_length_match', True))
        len_tolerance = int(params.get('len_tolerance', 0))
        target_lengths = []
        for w in self.words:
            sw = strip_punct(w)
            if sw:
                target_lengths.append(len(sw))
        target_exact_len = None
        target_preferred_len = None
        if target_lengths:
            unique_lens = sorted(set(target_lengths))
            if len(unique_lens) == 1:
                target_exact_len = unique_lens[0]
            else:
                target_preferred_len = int(round(sum(target_lengths) / float(len(target_lengths))))

        def candidate_length_ok(candidate):
            if not enforce_length_match:
                return True
            sc = strip_punct(candidate)
            if not sc:
                return False
            clen = len(sc)
            if target_exact_len is not None:
                return abs(clen - target_exact_len) <= len_tolerance
            if target_preferred_len is not None:
                return abs(clen - target_preferred_len) <= len_tolerance
            return True
        avoid=[]
        for word in self.words: # it's awkward if the distractor is the same as the real word
            avoid.append(strip_punct(word).lower())
            
        # load exact exclude words just in case a fallback reads from unfiltered sources
        global_exclude = set()
        exc_path = params.get('exclude_words', None)
        if exc_path and os.path.exists(exc_path):
            with open(exc_path, 'r', encoding='utf-8') as ef:
                for ln in ef:
                    ew = ln.strip()
                    if ew and not ew.startswith('#'):
                        global_exclude.add(ew.lower())

        # normalize banned list to lowercase for case-insensitive comparison
        banned_l = [b.lower() for b in banned]

        def candidate_surprisal(i, candidate):
            """Score candidate using full-context method when the model provides it."""
            if hasattr(model, 'get_surprisal_from_hidden'):
                try:
                    if i < len(self.hiddens) and self.hiddens[i] is not None:
                        return model.get_surprisal_from_hidden(self.hiddens[i], candidate)
                except Exception:
                    pass
            return model.get_surprisal(self.probs[i], candidate)
        def is_propn_candidate(candidate):
            if nlp_sp is None:
                return False
            key = strip_punct(candidate).lower()
            if key in _propn_cache:
                return _propn_cache[key]
            
            clean_cand = strip_punct(candidate)
            try:
                # Test the original (likely lowercase from dictionary)
                doc1 = nlp_sp(clean_cand)
                val1 = bool(doc1 and len(doc1) > 0 and doc1[0].pos_ == 'PROPN')
                
                val2 = False
                # If it wasn't flagged as PROPN but is lowercase, 
                # test its capitalized form. German SpaCy strictly relies on casing!
                if not val1 and clean_cand.islower() and lang == 'de':
                    doc2 = nlp_sp(clean_cand.capitalize())
                    val2 = bool(doc2 and len(doc2) > 0 and doc2[0].pos_ == 'PROPN')
                
                val = val1 or val2
            except Exception:
                val = False
            _propn_cache[key] = val
            return val
        def candidate_min_surprisal(candidate):
            """Minimum surprisal of candidate across all contexts for this label."""
            min_surp_val = float('inf')
            for i in range(len(self.probs)):
                dist_surp = candidate_surprisal(i, candidate)
                if dist_surp is None:
                    return None
                if dist_surp < min_surp_val:
                    min_surp_val = dist_surp
            return min_surp_val
        def pick_best_from_pool(pool, allow_banned=False, relax_length=False, enforce_pos=None):
            """Pick the most implausible candidate from a pool, respecting filters."""
            def _find_best(sub_pool):
                local_best = None
                local_best_surp = float('-inf')
                for dist in sub_pool:
                    dist_l = strip_punct(dist).lower()
                    if not dist_l:
                        continue
                    if (not relax_length) and (not candidate_length_ok(dist)):
                        continue
                    if dist_l in avoid or dist_l in global_exclude:
                        continue
                    if is_propn_candidate(dist):
                        continue
                    if (not allow_banned) and (dist_l in banned_l):
                        continue
                    if not re.match(r"^[A-Za-zÄÖÜäöüß\u0600-\u06FF]+$", strip_punct(dist)):
                        continue
                    s = candidate_min_surprisal(dist)
                    if s is None:
                        continue
                    if s > local_best_surp:
                        local_best_surp = s
                        local_best = dist
                return local_best, local_best_surp

            if target_pos and match_noun_pos:
                exact_pool = [c for c in pool if get_candidate_pos(c) == target_pos]
                best_cand, best_surp = _find_best(exact_pool)
                if best_cand is not None:
                    return best_cand, best_surp
                if target_is_noun: 
                    # Nouns MUST be nouns, no fallback allowed
                    return None, float('-inf')
            
            return _find_best(pool)

        def get_include_words_exact_len(length_value):
            """Optional fallback pool from include_words for strict exact-length matching."""
            include_path = params.get('include_words', None)
            if not include_path:
                return []
            if not os.path.exists(include_path):
                return []
            out = []
            seen = set()
            try:
                with open(include_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        w = line.strip()
                        if not w:
                            continue
                        if len(strip_punct(w)) != length_value:
                            continue
                        if not re.match(r"^[A-Za-zÄÖÜäöüß\u0600-\u06FF]+$", strip_punct(w)):
                            continue
                        wl = strip_punct(w).lower()
                        if wl in seen:
                            continue
                        seen.add(wl)
                        out.append(w)
            except Exception:
                return []
            random.shuffle(out)
            return out
        # initialize
        best_word = None
        best_min_surp = float('-inf')
        # For Mode B + early positions: separately track the best candidate
        # that also meets the boosted threshold, so the boost is effective.
        best_qualified_word = None
        best_qualified_surp = float('-inf')
        # When enabled, always pick the highest-surprisal candidate instead of
        # returning early on threshold satisfaction.
        force_max_surprisal = bool(params.get('force_max_surprisal', False))
        # Optional mode: try to match candidate mean surprisal to the target
        # mean surprisal of the real words. This is more precise than simple
        # thresholding and can be enabled with `params['match_surprisal']`.
        match_surprisal_mode = bool(params.get('match_surprisal', False)) and (not force_max_surprisal)
        if match_surprisal_mode and len(self.surprisals) > 0:
            # compute target mean surprisal for the real words in this label
            try:
                target_mean = sum(self.surprisals) / float(len(self.surprisals))
            except Exception:
                target_mean = None
            if target_mean is not None:
                def _find_best_match(sub_pool):
                    best_diff = float('inf')
                    best_cand = None
                    for dist in sub_pool:
                        dist_l = strip_punct(dist).lower()
                        if dist_l in banned_l or dist_l in avoid or dist_l in global_exclude:
                            continue
                        if not candidate_length_ok(dist):
                            continue
                        if is_propn_candidate(dist):
                            continue
                        # light continuation filter: skip if candidate literally contains the target
                        skip = False
                        for target in self.words:
                            t = strip_punct(target).lower()
                            if t and (t in dist_l or dist_l in t) and len(t) >= 4:
                                skip = True
                                break
                        if skip:
                            continue
                        # compute mean surprisal of this candidate across all sentences
                        ssum = 0.0
                        count = 0
                        for i in range(len(self.probs)):
                            try:
                                dist_surp = candidate_surprisal(i, dist)
                            except Exception:
                                dist_surp = None
                            if dist_surp is None:
                                continue
                            ssum += dist_surp
                            count += 1
                        if count == 0:
                            continue
                        mean_surp = ssum / float(count)
                        diff = abs(mean_surp - target_mean)
                        if diff < best_diff:
                            best_diff = diff
                            best_cand = dist
                    return best_cand

                best_candidate = None
                if target_pos and match_noun_pos:
                    exact_pool = [c for c in distractor_opts if get_candidate_pos(c) == target_pos]
                    best_candidate = _find_best_match(exact_pool)
                    
                    if not best_candidate and not target_is_noun:
                        best_candidate = _find_best_match(distractor_opts)
                else:
                    best_candidate = _find_best_match(distractor_opts)

                if best_candidate:
                    # Apply grammatical casing based on the DISTRACTOR'S class
                    self.distractor = _get_german_grammatical_case(best_candidate, dictionary)
                    return self.distractor

        # 1. Pre-filter candidates (cheap checks: length, banned, POS, repeat)
        qualified_candidates = []
        for dist in distractor_opts:
            dist_l = strip_punct(dist).lower()
            if dist_l in banned_l or dist_l in avoid or dist_l in global_exclude:
                continue
            if not candidate_length_ok(dist):
                continue
            if is_propn_candidate(dist) and params.get("exclude_propn_candidates", False):
                continue
            # light continuation filter
            skip = False
            for target in self.words:
                t = strip_punct(target).lower()
                if t and (t in dist_l or dist_l in t) and len(t) >= 4:
                    skip = True
                    break
            if skip:
                continue
            qualified_candidates.append(dist)
            
        if not qualified_candidates:
            # handle empty pool via fallback logic below
            target_rep = self.words[0] if self.words else "???"
            print(f"  [Batch] Scoring 0 candidates for target '{target_rep}'...")
        else:
            # 2. Batch score the qualified pool for all sentence contexts
            target_rep = self.words[0] if self.words else "???"
            print(f"  [Batch] Scoring {len(qualified_candidates)} candidates for target '{target_rep}'...")
            
            candidate_scores = []
            for j in range(len(self.probs)):
                hidden = self.hiddens[j]
                # --- CUDA OOM PREVENTION ---
                # Now set to 256 for maximum throughput on the memory-efficient engine.
                chunk_size = 512
                all_scores = []
                for i in range(0, len(qualified_candidates), chunk_size):
                    chunk = qualified_candidates[i : i + chunk_size]
                    try:
                        chunk_scores = model.get_surprisal_batch_from_hidden(hidden, chunk, batch_size=512)
                        all_scores.extend(chunk_scores)
                    except Exception as e:
                        logging.error(f"Batch scoring failed for chunk {i}: {e}")
                        # Fallback to slower single scoring for this chunk if GPU fails
                        for c in chunk:
                            all_scores.append(model.get_surprisal(self.probs[j], c))
                
                # Map them back to the words
                candidate_scores.append(dict(zip(qualified_candidates, all_scores)))
                
            # 3. Iterate through candidates and apply surprisal filters/selection
            for dist in qualified_candidates:
                # Get the min surprisal across contexts (min of surprisals = worst case)
                vals = [candidate_scores[j][dist] for j in range(len(self.probs))]
                min_surp_val = min(vals)
                
                # Check if it meets all thresholds
                meets_all = True
                for j in range(len(self.probs)):
                    if candidate_scores[j][dist] < self.surprisal_targets[j]:
                        meets_all = False
                        break
                
                # Update best_word trackers
                if meets_all and min_surp_val > best_min_surp:
                    best_min_surp = min_surp_val
                    best_word = dist

                if force_max_surprisal and is_early and early_boost > 0:
                    if meets_all and min_surp_val > best_qualified_surp:
                        best_qualified_surp = min_surp_val
                        best_qualified_word = dist

                if not force_max_surprisal and meets_all:
                    # Apply grammatical casing based on the DISTRACTOR'S class
                    best_word = dist
                    break
        # Mode B + early position boost: prefer the best candidate that meets
        # the boosted threshold.  Falls back to the unconstrained best_word if
        # nothing passed the threshold.
        if best_qualified_word is not None:
            best_word = best_qualified_word
            best_min_surp = best_qualified_surp
        # Hard guarantee: never return x-x-x for non-initial positions.
        # If no candidate survived strict filters, relax constraints in stages.
        if best_word is None:
            allow_banned_fallback = bool(params.get("allow_banned_fallback", False))
            fallback = list(distractor_opts)
            random.shuffle(fallback)
            cand, cand_surp = pick_best_from_pool(fallback, allow_banned=False)
            if cand is None and allow_banned_fallback:
                cand, cand_surp = pick_best_from_pool(fallback, allow_banned=True)
            desired_len = target_exact_len if target_exact_len is not None else target_preferred_len
            if cand is None and enforce_length_match and desired_len is not None:
                # Strict exact-length fallback from full in-memory dictionary first.
                try:
                    exact_full_pool = [w.text for w in getattr(dict, 'words', []) if len(strip_punct(w.text)) == desired_len]
                except Exception:
                    exact_full_pool = []
                if exact_full_pool:
                    random.shuffle(exact_full_pool)
                    cand, cand_surp = pick_best_from_pool(exact_full_pool, allow_banned=False)
                    if cand is None and allow_banned_fallback:
                        cand, cand_surp = pick_best_from_pool(exact_full_pool, allow_banned=True)
            if cand is None and enforce_length_match and desired_len is not None:
                # Then try include_words (can contain short forms filtered from main dict).
                include_pool = get_include_words_exact_len(desired_len)
                if include_pool:
                    cand, cand_surp = pick_best_from_pool(include_pool, allow_banned=False)
                    if cand is None and allow_banned_fallback:
                        cand, cand_surp = pick_best_from_pool(include_pool, allow_banned=True)
            if cand is None:
                cand, cand_surp = pick_best_from_pool(fallback, allow_banned=False, relax_length=True)
                if cand is None and allow_banned_fallback:
                    cand, cand_surp = pick_best_from_pool(fallback, allow_banned=True, relax_length=True)
            if cand is None:
                try:
                    # Fix: use 'dictionary' instead of 'dict'. Enforce POS guard to prevent cross-contamination.
                    def emergency_pos_ok(w_pos):
                        if not match_noun_pos: return True
                        return (w_pos == 'NOUN' or w_pos == 'PROPN') if target_is_noun else (w_pos != 'NOUN' and w_pos != 'PROPN')
                    
                    target_len = target_exact_len or desired_len or 5
                    emergency_pool = [w.text for w in getattr(dictionary, 'words', []) 
                                      if emergency_pos_ok(getattr(w, 'pos', None))
                                      and abs(len(w.text) - target_len) <= (len_tol if enforce_length_match else 10)]
                except Exception:
                    emergency_pool = []
                if emergency_pool:
                    sample_n = min(800, len(emergency_pool))
                    fallback_emerg = random.sample(emergency_pool, sample_n)
                    cand, cand_surp = pick_best_from_pool(fallback_emerg, allow_banned=False)
                    if cand is None and allow_banned_fallback:
                        cand, cand_surp = pick_best_from_pool(fallback_emerg, allow_banned=True)
                    if cand is None:
                        cand, cand_surp = pick_best_from_pool(fallback_emerg, allow_banned=False, relax_length=True)
                        if cand is None and allow_banned_fallback:
                            cand, cand_surp = pick_best_from_pool(fallback_emerg, allow_banned=True, relax_length=True)
            if cand is None:
                # Final defensive fallback: force a real word token.
                cand = "wort"
                cand_surp = float('-inf')
            best_word = cand
            best_min_surp = cand_surp
        if best_word is None:
            # Last ditch attempt: ignore POS constraints entirely
            fallback_pool = list(distractor_opts)
            random.shuffle(fallback_pool)
            # Find the absolute best available word regardless of threshold
            cand, cand_surp = pick_best_from_pool(fallback_pool, allow_banned=False, relax_length=True)
            if cand is None:
                cand, cand_surp = pick_best_from_pool(fallback_pool, allow_banned=True, relax_length=True)
            
            if cand is not None:
                best_word = cand
                best_min_surp = cand_surp

        if best_word is None:
            # If still nothing, pick ANY real word of correct length from dictionary
            try:
                # Avoid 'wort' if any other word exists in the dictionary pool
                emergency_pool = [w.text for w in getattr(dictionary, 'words', []) 
                                if abs(len(w.text) - (target_exact_len or 5)) <= 2 
                                and w.text.lower() not in banned_l]
                if emergency_pool:
                    random.shuffle(emergency_pool)
                    for w_text in emergency_pool:
                        if not match_noun_pos:
                            best_word = w_text
                            break
                        
                        is_n = dictionary.has_titlecase_variant(w_text)
                        if target_is_noun and is_n:
                            best_word = w_text
                            break
                        elif not target_is_noun and not is_n:
                            best_word = w_text
                            break
                
                if best_word is None:
                    best_word = "wort"
            except Exception:
                best_word = "wort"
                
            best_min_surp = float('-inf')

        # Final casing pass based on DISTRACTOR category
        self.distractor = _get_german_grammatical_case(best_word, dictionary)

        if not force_max_surprisal:
            logging.warning("Could not find a word to meet threshold for item %s, label %s, returning %s with %f min surp instead",
                self.id, self.lab, self.distractor, best_min_surp)
        return self.distractor


class Sentence_Set:
    """A set of sentence objects, with the same id"""

    def __init__(self, id):
        self.id = id
        self.sentences = []
        self.label_ids = set()
        self.first_labels = set()
        self.labels = {}  # dictionary of label:label object

    def add(self, sentence):
        """Adds a sentence item to the sentence_set"""
        if sentence.id == self.id:
            self.sentences.append(sentence)
            first_label = sentence.labels[0]
            self.first_labels = self.first_labels.union(set([first_label]))
            self.label_ids = self.label_ids.union(sentence.labels[1:])
            if self.first_labels & self.label_ids:
                logging.error("Labels of first words cannot match labels of later words in the same set in item %s", self.id)
                raise ValueError()
        else:
            logging.error("ID doesn't match")
            raise ValueError()

    def do_model(self, model):
        """Applies model to sentences"""
        for sentence in self.sentences:
            sentence.do_model(model)

    def do_surprisals(self, model):
        """Gets surprisals for the real words"""
        for sentence in self.sentences:
            sentence.do_surprisal(model)

    def make_labels(self, params=None):
        """Regroups the stuff in the sentence items into by-label groups"""
        params = params or {}
        lang = _detect_language(params)
        # SpaCy-only POS tagging; else no POS
        nlp_sp = _get_spacy_nlp(lang)
        if nlp_sp is not None:
            for sentence in self.sentences:
                try:
                    # Use the words list directly as tokens for alignment
                    from spacy.tokens import Doc
                    doc = Doc(nlp_sp.vocab, words=sentence.words)
                    # Run the tagger/parser on the pre-tokenized doc
                    for name, proc in nlp_sp.pipeline:
                        if name not in ["tok2vec", "tagger", "attribute_ruler", "lemmatizer"]:
                            # We only strictly need the tagger for POS
                            if name != "tagger" and name != "attribute_ruler":
                                continue
                        doc = proc(doc)
                    pos_tags = [t.pos_ for t in doc]
                    sentence.pos_tags = pos_tags
                except Exception:
                    sentence.pos_tags = None
        else:
            for sentence in self.sentences:
                sentence.pos_tags = None

        # --- DYNAMIC LENGTH ALIGNMENT FIX (ORDERED) ---
        new_label_ids = []
        seen_labs = set()
        for sentence in self.sentences:
            for i in range(1, len(sentence.labels)):
                orig_lab = sentence.labels[i]
                target_len = len(strip_punct(sentence.words[i]))
                if target_len > 0:
                    new_lab = f"{orig_lab}_L{target_len}"
                else:
                    new_lab = orig_lab
                
                if new_lab not in seen_labs:
                    new_label_ids.append(new_lab)
                    seen_labs.add(new_lab)
                
                if new_lab != orig_lab:
                    sentence.labels[i] = new_lab
                    if orig_lab in sentence.probs:
                        sentence.probs[new_lab] = sentence.probs.pop(orig_lab)
                    if hasattr(sentence, 'hiddens') and orig_lab in sentence.hiddens:
                        sentence.hiddens[new_lab] = sentence.hiddens.pop(orig_lab)
                    if hasattr(sentence, 'surprisal') and orig_lab in sentence.surprisal:
                        sentence.surprisal[new_lab] = sentence.surprisal.pop(orig_lab)
                
        self.label_ids = new_label_ids
        # ------------------------------------

        for lab in self.label_ids: #init label objects
            self.labels[lab] = Label(self.id, lab)
        for sentence in self.sentences: #dump stuff into the label objects
            for i in range(1, len(sentence.labels)):
                lab = sentence.labels[i]
                hidden = sentence.hiddens.get(lab) if hasattr(sentence, 'hiddens') else None
                # if POS tags were computed, pass the POS for this token to the label
                pos = None
                if hasattr(sentence, 'pos_tags') and sentence.pos_tags is not None:
                    try:
                        pos = sentence.pos_tags[i]
                    except Exception:
                        pos = None
                self.labels[lab].add_sentence(sentence.words[i], sentence.probs[lab], sentence.surprisal[lab], hidden)
                # store pos into the last appended slot
                try:
                    self.labels[lab].pos[-1] = pos
                except Exception:
                    pass

    def do_distractors(self, model, d, threshold_func, params, repeats):
        """Get distractors using specified stuff"""
        lang = _detect_language(params)
        banned = repeats.banned[:] #don't allow duplicate distractors within the set
        for label in self.labels.values(): #get distractors for each label
            dist = label.choose_distractor(model, d, threshold_func, params, banned)
            banned.append(dist)
            repeats.increment(dist)

        def choose_first_word(sentence):
            """Choose a real-word distractor for sentence-initial position.
            Enforce exact stripped-length match whenever possible.
            """
            target = strip_punct(sentence.words[0])
            target_l = target.lower()
            target_len = len(target)
            banned_l = set([strip_punct(b).lower() for b in banned])

            min_length, max_length, min_freq, max_freq = threshold_func([sentence.words[0]], params)
            # detect if target is a noun
            target_is_noun_first = False
            if hasattr(sentence, 'pos_tags') and sentence.pos_tags and sentence.pos_tags[0] in ('NOUN', 'PROPN'):
                target_is_noun_first = True
            
            pos_filter = None
            if params.get('match_noun_pos', False):
                pos_filter = 'NOUN' if target_is_noun_first else '!NOUN'
            
            print(f"  [First] Finding first-word distractor for '{target}'...")
            opts = d.get_potential_distractors(min_length, max_length, min_freq, max_freq, params, pos_filter=pos_filter)
            
            if pos_filter == '!NOUN':
                num_req = int(params.get('num_to_test', 100))
                if len(opts) < (num_req // 2):
                    fallback_opts = d.get_potential_distractors(min_length, max_length, min_freq, max_freq, params, pos_filter=None)
                    existing = set(opts)
                    for opt in fallback_opts:
                        if opt not in existing:
                            opts.append(opt)
                            
            random.shuffle(opts)
            for cand in opts:
                base = strip_punct(cand)
                if (not base) or (not re.match(r"^[A-Za-zÄÖÜäöüß\u0600-\u06FF]+$", base)):
                    continue
                if len(base) != target_len:
                    continue
                low = base.lower()
                if low == target_l or low in banned_l:
                    continue
                return cand

            # Fallback: search full dictionary pool for closest valid match.
            best = None
            best_diff = 10**9
            try:
                pool = [w.text for w in getattr(d, "words", [])]
            except Exception:
                pool = []
            random.shuffle(pool)
            for cand in pool:
                base = strip_punct(cand)
                if (not base) or (not re.match(r"^[A-Za-zÄÖÜäöüß\u0600-\u06FF]+$", base)):
                    continue
                low = base.lower()
                if low == target_l or low in banned_l:
                    continue
                diff = abs(len(base) - target_len)
                if diff < best_diff:
                    best = cand
                    best_diff = diff
                    if diff == 0:
                        break
            return best if best is not None else "wort"

        use_first_placeholder = bool(params.get("first_token_placeholder", True))
        for sentence in self.sentences: #give the sentences the distractors
            sentence.distractors = []
            if use_first_placeholder:
                first_len = len(strip_punct(sentence.words[0]))
                first_placeholder = _placeholder_for_length(first_len)
                sentence.distractors.append(_copy_edge_punct_no_case(sentence.words[0], first_placeholder))
            else:
                first_dist = choose_first_word(sentence)
                first_dist = copy_punct(sentence.words[0], first_dist)
                if lang == 'en':
                    first_dist = _normalize_english_distractor_case(sentence.words[0], first_dist)
                sentence.distractors.append(first_dist)
            for i in range(1, len(sentence.labels)):
                lab = sentence.labels[i]
                # we match distractors to their real words on punctuation
                distractor = copy_punct(sentence.words[i], self.labels[lab].distractor)
                if lang == 'en':
                    distractor = _normalize_english_distractor_case(sentence.words[i], distractor)
                sentence.distractors.append(distractor)
            # (No forced pseudoword replacements: system uses real-word distractors only)
            # Post-process casing: only apply when appropriate (e.g., German pipeline).
            # Allow explicit override via `params['apply_postcase']` (True/False).
            try:
                apply_postcase = params.get('apply_postcase', None)
            except Exception:
                apply_postcase = None
            if apply_postcase is None:
                # Auto-detect: prefer post-case when model_loc or dictionary_class mentions 'german'
                try:
                    model_loc = (params.get('model_loc', '') or '').lower()
                    dict_cls = (params.get('dictionary_class', '') or '').lower()
                    hf_name = (params.get('hf_model_name', '') or '').lower()
                    apply_postcase = ('german' in model_loc) or ('german' in dict_cls) or ('dbmdz' in hf_name)
                except Exception:
                    apply_postcase = False
            # Read match_target_case param
            match_target_case = bool(params.get('match_target_case', False))
            if apply_postcase and not match_target_case:
                try:
                    for j in range(len(sentence.distractors)):
                        # Use the original word's POS tag (from full-sentence
                        # spaCy tagging in make_labels) instead of re-tagging
                        # the distractor in isolation.
                        src_pos = None
                        if hasattr(sentence, 'pos_tags') and sentence.pos_tags is not None:
                            try:
                                src_pos = sentence.pos_tags[j]
                            except (IndexError, TypeError):
                                src_pos = None
                        sentence.distractors[j] = _normalize_distractor_token(
                            sentence.distractors[j], d, lang=lang, source_pos=src_pos)
                except Exception:
                    pass
            # match_target_case: copy the target word's casing onto the distractor
            if match_target_case:
                for j in range(len(sentence.distractors)):
                    if j >= len(sentence.words):
                        break
                    if _is_x_placeholder_token(sentence.distractors[j]):
                        continue
                    target_tok = sentence.words[j]
                    dist_tok = sentence.distractors[j]
                    _, t_body, _ = _split_punct(target_tok)
                    d_prefix, d_body, d_suffix = _split_punct(dist_tok)
                    if not d_body or not t_body:
                        continue
                    if t_body.isupper():
                        new_body = d_body.upper()
                    elif t_body[0].isupper():
                        new_body = d_body[0:1].upper() + d_body[1:].lower()
                    else:
                        new_body = d_body.lower()
                    sentence.distractors[j] = d_prefix + new_body + d_suffix
            # Keep first placeholder shape stable (no auto-capitalization side effects).
            if use_first_placeholder and sentence.distractors:
                first_len = len(strip_punct(sentence.words[0]))
                sentence.distractors[0] = _copy_edge_punct_no_case(
                    sentence.words[0], _placeholder_for_length(first_len)
                )
            sentence.distractor_sentence = " ".join(sentence.distractors) #and in sentence_format

    def clean_up(self):
        """Removes memory intensive things like label items and prob distributions"""
        self.labels = {}
        for sentence in self.sentences:
            sentence.probs = {}
