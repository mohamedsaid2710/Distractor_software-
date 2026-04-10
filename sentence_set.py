import logging
from utils import copy_punct, strip_punct
from limit_repeats import Repeatcounter
import re
import random
import os
try:
    import wordfreq as _wordfreq_mod
except ImportError:
    _wordfreq_mod = None

print("\n" + "="*60)
print("GERMAN CASING V2.2: DUAL-CHECK + SENTENCE-FRAME TAGGING ACTIVE")
print("="*60 + "\n")

# Lazy import for semantic filtering
_semantic_filter_module = None

def _get_semantic_filter():
    """Lazy-load semantic filter module."""
    global _semantic_filter_module
    if _semantic_filter_module is None:
        try:
            import semantic_filter
            _semantic_filter_module = semantic_filter
        except ImportError:
            _semantic_filter_module = False  # Mark as unavailable
    return _semantic_filter_module if _semantic_filter_module else None

# Common uppercase acronyms to preserve when used as distractors in English.
_EN_ACRONYM_WHITELIST = {
    'ai', 'api', 'bbc', 'cia', 'cpu', 'eu', 'fbi', 'gdp', 'gps', 'gpu',
    'ibm', 'imf', 'ml', 'nasa', 'nato', 'oecd', 'uk', 'un', 'usa', 'uefa',
    'who',
}

_X_PLACEHOLDER_RE = re.compile(r"^x(?:-x)*$", re.IGNORECASE)

# lazy NLP pipeline (initialized on first use)
_nlp_model = {}

def _get_nlp_model(lang='de', params=None):
    """Return a loaded NLP pipeline. Stanza for 'de', SpaCy for 'en', None for others."""
    global _nlp_model
    lang_lower = str(lang or '').lower()
    
    if lang_lower.startswith('en'):
        key = 'en'
    elif lang_lower.startswith('de'):
        key = 'de'
    elif lang_lower.startswith('ar'):
        # For Arabic, POS tagging is done exclusively via Farasa inside wordfreq_distractor
        # We return a dummy object just so `nlp_model is None` checks pass, allowing
        # get_candidate_pos to read from dictionary.pos_cache
        return "FARASA_DELEGATE"
    else:
        return None
        
    if key in _nlp_model:
        return _nlp_model[key]
        
    if key == 'de':
        try:
            import stanza
            # Use GPU if available, default to True
            use_gpu = True
            if params is not None:
                use_gpu = str(params.get('use_gpu', True)).lower() in ('true', '1')
            
            try:
                # Try to load; download if necessary
                _nlp_model[key] = stanza.Pipeline('de', processors='tokenize,mwt,pos,lemma', use_gpu=use_gpu)
            except Exception:
                stanza.download('de')
                _nlp_model[key] = stanza.Pipeline('de', processors='tokenize,mwt,pos,lemma', use_gpu=use_gpu)
            return _nlp_model[key]
        except ImportError:
            logging.error("Stanza not found. Please install with: pip install stanza")
            _nlp_model[key] = None
            return None
    
    elif key == 'en':
        try:
            import spacy
        except Exception:
            _nlp_model[key] = None
            return None

        model_names = ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm']
        for model_name in model_names:
            try:
                _nlp_model[key] = spacy.load(model_name)
                return _nlp_model[key]
            except Exception:
                continue
        _nlp_model[key] = None
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
    NOTE: This global version is a safe fallback. The real logic lives in the
    local is_propn_candidate() defined inside choose_distractor(), which has
    access to the dictionary's pos_cache. This global version should rarely
    be reached, but must not crash.
    """
    clean_cand = strip_punct(cand)
    if not clean_cand:
        return False
        
    key = clean_cand.lower()
    if key in PROPN_CACHE:
        return PROPN_CACHE[key]
    
    # Without access to a dictionary object, we cannot reliably determine PROPN.
    # Return False (safe default: treat as non-PROPN and let downstream filters handle it).
    return False



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


def _normalize_english_distractor_case(distractor_token, is_first_word=False, target_token=""):
    """English-specific casing normalization based on the distractor's properties and target's case."""
    if _is_x_placeholder_token(distractor_token):
        return distractor_token
    d_prefix, d_body, d_suffix = _split_punct(distractor_token)
    if not d_body:
        return distractor_token

    # Interactive Casing: explicitly copy uppercase or titlecase from the target word if available
    if target_token:
        _, t_body, _ = _split_punct(target_token)
        if t_body:
            if t_body.isupper() and len(t_body) > 1:
                return d_prefix + d_body.upper() + d_suffix
            elif t_body.istitle() or (len(t_body) == 1 and t_body.isupper()):
                return d_prefix + d_body.capitalize() + d_suffix

    # Capitalize known acronym distractors.
    if d_body.lower() in _EN_ACRONYM_WHITELIST:
        return d_prefix + d_body.upper() + d_suffix

    # Sentence-initial capitalization
    if is_first_word and d_body:
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
    if ('german' in model_loc) or ('german' in dict_cls) or ('benjamin' in hf_name) or ('gerpt2' in hf_name):
        return 'de'
    if ('arabic' in model_loc) or ('arabic' in dict_cls) or ('aragpt' in hf_name) or ('aubmindlab' in hf_name):
        return 'ar'
    return 'en'


# ---------------------------------------------------------------------------
# POS-aware fallback cascade for non-noun targets.
# When the target is not a noun, try exact POS match first, then compatible
# word classes, and only allow a NOUN distractor as the absolute last resort.
# ---------------------------------------------------------------------------
_POS_COMPATIBLE_CLASSES = {
    'VERB': ['VERB', 'ADV', 'ADJ', 'ADP'],
    'ADJ':  ['ADJ', 'ADV', 'VERB', 'ADP'],
    'ADV':  ['ADV', 'ADJ', 'VERB', 'ADP'],
    'ADP':  ['ADP', 'ADV', 'ADJ', 'VERB'],
    'DET':  ['DET', 'ADP', 'ADV', 'ADJ'],
    'PRON': ['PRON', 'DET', 'ADP', 'ADV'],
    'PART': ['PART', 'ADV', 'ADP', 'ADJ'],
    'CCONJ': ['CCONJ', 'SCONJ', 'ADP', 'ADV'],
    'SCONJ': ['SCONJ', 'CCONJ', 'ADP', 'ADV'],
}


def _get_german_grammatical_case(token, dict_obj, is_first_word=False, target_token="", match_casing_only=False):
    """Determine correct German casing based strictly on the distractor's own POS.

    Uses the dictionary's pos_cache (built offline via precompute_pos.py or
    in batches by SpaCy) to enforce grammatical capitalization solely on
    whether the generated distractor itself is a NOUN/PROPN.

    When match_casing_only=True, bypasses POS-cache lookup and instead
    mirrors the target word's capitalisation.  This is safe because the
    candidate pool was already filtered (NOUN for uppercase targets,
    !NOUN for lowercase targets).

    Rules (German orthography):
      - Nouns (NOUN/PROPN) → Titlecase
      - Everything else     → lowercase
      - Sentence-initial     → capitalize first letter regardless of POS
    """
    if _is_x_placeholder_token(token):
        return token
    prefix, body, suffix = _split_punct(token)
    if not body:
        return token

    # --- CASING-ONLY MODE: apply correct casing based on the distractor's own class ---
    # The German grammar guard in get_potential_distractors already ensured that
    # nouns do not appear in non-noun slots (and vice-versa) when match_casing_only
    # is active.  Here we simply apply the correct TitleCase / lowercase to whatever
    # distractor was selected, using pos_cache as the authoritative source.

    # When match_casing_only=True: mirror the target token's casing (already guarded by grammar)
    # When False: apply grammatical rules (nouns titlecase, others lowercase)
    if match_casing_only and target_token:
        # Casing-only mode: just mirror target casing
        target_is_cap = target_token[0].isupper() if target_token else False
        new_body = body[0].upper() + body[1:].lower() if target_is_cap else body.lower()
    else:
        # Standard German grammar: determine if *distractor* is a noun and apply rules
        distractor_is_noun = False
        t_lower = body.lower()
        
        if hasattr(dict_obj, 'pos_cache') and t_lower in dict_obj.pos_cache:
            distractor_is_noun = dict_obj.pos_cache[t_lower] in ('NOUN', 'PROPN')
        elif hasattr(dict_obj, 'get_titlecase_variant'):
            try:
                tv = dict_obj.get_titlecase_variant(body)
                distractor_is_noun = isinstance(tv, str)
            except Exception:
                pass

        if distractor_is_noun:
            new_body = body[0].upper() + body[1:].lower()
        else:
            new_body = body.lower()

    if is_first_word and new_body:
        new_body = new_body[0].upper() + new_body[1:]

    # Strip exact trailing hyphens as requested by user, while keeping suffix (.,!)
    new_body = new_body.rstrip("-")
    return prefix + new_body + suffix

def _normalize_distractor_token(token, dict_obj, lang='de', is_first_word=False, target_token="", match_casing_only=False):
    """Normalize casing for a single distractor token by its own grammatical POS."""
    if lang == 'de':
        return _get_german_grammatical_case(token, dict_obj, is_first_word=is_first_word,
                                            target_token=target_token, match_casing_only=match_casing_only)
    
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
        match_casing_only = bool(params.get('match_casing_only', False))
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

        # --- Short word boost ---
        # For very short words (≤3 chars), require higher surprisal since
        # short function words have many plausible alternatives
        short_word_boost = float(params.get('short_word_boost', 0))
        short_word_max_len = int(params.get('short_word_max_len', 3))
        is_short_word = False
        if self.words:
            # Check if any target word is short
            for w in self.words:
                clean_w = strip_punct(w)
                if clean_w and len(clean_w) <= short_word_max_len:
                    is_short_word = True
                    break
        
        for surprisal in self.surprisals:  # calculate desired surprisal thresholds
            if absolute_threshold_only:
                base = params["min_abs"]
            else:
                base = max(params["min_abs"], surprisal + params["min_delta"])
            if is_early:
                base += early_boost
            if is_short_word:
                base += short_word_boost
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
        nlp_model = _get_nlp_model(lang, params=params)

        _propn_cache = {}
        _pos_cache = {}
        
        def get_candidate_pos(candidate, target_noun_context=False):
            if nlp_model is None: return None
            clean = strip_punct(candidate)
            if not clean: return None
            
            # FAST PATH: Check dictionary global cache first for zero-overhead
            low_key = clean.lower()
            if hasattr(dictionary, 'pos_cache') and low_key in dictionary.pos_cache:
                spacy_pos = dictionary.pos_cache[low_key]
                return spacy_pos
            
            if target_noun_context:
                clean = clean.capitalize()
                
            key = clean
            if key in _pos_cache: return _pos_cache[key]
            try:
                if nlp_model == "FARASA_DELEGATE":
                    if hasattr(dictionary, 'nlp_sp') and dictionary.nlp_sp is not None:
                        tagged = dictionary.nlp_sp.tag(clean)
                        parts = tagged.split('+')
                        val = 'X'
                        for part in reversed(parts):
                            if '/' in part:
                                curr = part.split('/')[-1].strip()
                                if curr not in ('CONJ', 'PREP', 'DET', 'PART', 'PUNC'):
                                    val = curr
                                    break
                        if val == 'X' and parts and '/' in parts[-1]:
                            val = parts[-1].split('/')[-1].strip()
                        if val == 'NOUN': val = 'NOUN'
                        elif val == 'ADJ': val = 'ADJ'
                        elif val == 'V': val = 'VERB'
                        elif 'PRON' in val: val = 'PRON'
                    else:
                        val = None
                elif lang == 'de':
                    # Stanza
                    doc = nlp_model(clean)
                    val = doc.sentences[0].words[0].upos if doc.sentences and doc.sentences[0].words else None
                else:
                    doc = nlp_model(clean)
                    val = doc[0].pos_ if doc and len(doc) > 0 else None
            except Exception:
                val = None
            _pos_cache[key] = val
            return val

        target_pos = get_candidate_pos(self.words[0]) if self.words else None
        
        # --- CASING CALCULATION ---
        sw_target = strip_punct(self.words[0]) if self.words else ""
        target_is_lower = sw_target[0].islower() if sw_target else True

        if lang == 'de' and match_casing_only and not match_noun_pos:
            # Casing-only mode (no POS matching): use capitalisation as a proxy
            # for German nounness.  Safe shortcut: German nouns are always TitleCase.
            target_is_noun = not target_is_lower
        elif not match_casing_only:
            # Pure POS mode or non-German: derive nounness from the actual POS tag.
            target_is_noun = (target_pos == 'NOUN')
        # When BOTH match_casing_only AND match_noun_pos are True:
        # Keep the POS-derived target_is_noun that was already computed above
        # (lines 440-468 via self.pos / dictionary.has_titlecase_variant).
        # match_casing_only handles casing independently via target_is_capitalized
        # (line 555); match_noun_pos handles POS via pos_filter (lines 542-549).
        # The two flags are fully orthogonal and must not overwrite each other.

        # --- POS FALLBACK CASCADE ---
        # For noun targets: NOUN candidates only.
        # For non-noun targets: build a preference list from
        # _POS_COMPATIBLE_CLASSES so nouns are the absolute last resort.
        compatible_pos_list = None  # ordered list for non-noun cascade
        pos_filter = None
        if match_noun_pos:
            if target_is_noun:
                pos_filter = 'NOUN'
            else:
                pos_filter = '!NOUN'
                # Build ordered preference list for later cascade sorting
                compatible_pos_list = _POS_COMPATIBLE_CLASSES.get(
                    target_pos, ['ADV', 'ADJ', 'VERB', 'ADP'])

        # Ironclad Casing: Inject target casing into params for this word
        orig_target = self.words[0] if self.words else ""
        target_stripped = strip_punct(orig_target)
        target_is_cap = target_stripped[0].isupper() if target_stripped else False
        params['target_is_capitalized'] = target_is_cap

        # PRE-TAG TARGET WORD: Tag target *before* finding distractors
        # This ensures has_titlecase_variant() can find target POS in cache for accurate detection
        if hasattr(dictionary, 'batch_tag_words'):
            target_words_to_tag = [target_stripped.lower()] if target_stripped else []
            dictionary.batch_tag_words(target_words_to_tag, params=params)

        print(f"  [Batch] Finding distractors for '{orig_target}' (Cap: {target_is_cap})...")
        min_length, max_length, min_freq, max_freq = threshold_func(self.words, params)
        distractor_opts = dictionary.get_potential_distractors(min_length, max_length, min_freq, max_freq, params, pos_filter=pos_filter)
        # --- ADAPTIVE LENGTH SEARCH (Fix for 0-candidate long words) ---
        # For Nouns, we PREFER shorter, actual nouns rather than long adjectives
        # that barely fit the length profile. 
                
        if not distractor_opts:
            for extra_len in range(1, 11): # Try widening up to +/- 10 characters
                logging.info(f"0 candidates for {self.words} after primary fetch. Widening search by +/- {extra_len}")
                # Bias search downwards: it's better to have a shorter noun than no noun
                min_len_search = max(2, min_length - extra_len)
                max_len_search = max_length + (extra_len // 2) if target_is_noun else (max_length + extra_len)
                
                distractor_opts = dictionary.get_potential_distractors(
                    min_len_search, max_len_search, 
                    min_freq, max_freq, params, pos_filter=pos_filter
                )
                if distractor_opts:
                    break                
                # --- FINAL FALLBACK: Frequency Relaxation ---
                # If still nothing, try ignoring frequency thresholds for this length widening
                logging.info(f"Still 0 candidates. Relaxing frequency constraints for +/- {extra_len}")
                distractor_opts = dictionary.get_potential_distractors(
                    min_len_search, max_len_search, 
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

        # Only skip the POS cascade in pure casing-only mode (match_noun_pos=False).
        # When match_noun_pos is active the cascade must always run regardless of
        # casing mode — skipping it would silently disable POS matching.
        skip_cascade = (lang == 'de' and match_casing_only and not match_noun_pos and not target_is_noun)
        if match_noun_pos and target_pos and not skip_cascade:
            exact = []
            compatible = []  # candidates matching a compatible POS class
            noun_fallback = []  # NOUN candidates — absolute last resort for non-noun targets
            others = []
            bad_pos = {'PROPN', 'X', 'SYM', 'INTJ', 'NUM', 'PUNCT'}

            # For non-noun targets, NOUNs go into noun_fallback, not rejected.
            # They are only used if exact + compatible are both exhausted.
                
            # --- HYPER-SPEED CPU FIX: BATCH TAGGING WITH NATIVE SENTENCE CONTEXT ---
            # --- HYPER-SPEED GPU FIX: BATCH TAGGING WITH NATIVE SENTENCE CONTEXT ---
            try:
                if nlp_model is not None and distractor_opts:
                    if hasattr(self, 'context_words') and hasattr(self, 'target_idx') and self.target_idx >= 0 and lang != 'de':
                        # Spacy only for now on native context (Stanza natively modifies sentence strings so it's harder in batch)
                        from spacy.tokens import Doc
                        docs = []
                        valid_pairs = []
                        cw = self.context_words[:]
                        idx = self.target_idx
                        for c in distractor_opts:
                            clean = strip_punct(c)
                            if not clean: continue
                            if target_is_noun:
                                clean = clean.capitalize()
                            cw[idx] = clean
                            # Create a Doc natively in the exact original sentence context!
                            docs.append(Doc(nlp_model.vocab, words=cw))
                            valid_pairs.append(c)
                            
                        docs_parsed = list(nlp_model.pipe(docs, disable=['ner', 'parser', 'lemmatizer'], batch_size=int(params.get('nlp_batch_size', params.get('spacy_batch_size', 2000)))))
                        for c, doc in zip(valid_pairs, docs_parsed):
                            c_pos = doc[idx].pos_ if len(doc) > idx else None
                            is_exact_match = (c_pos == target_pos) or (target_is_noun and c_pos == 'PROPN')
                            
                            if is_exact_match:
                                exact.append(c)
                            elif not target_is_noun and c_pos == 'NOUN':
                                noun_fallback.append(c)
                            elif compatible_pos_list and c_pos in compatible_pos_list:
                                compatible.append(c)
                            elif c_pos not in bad_pos:
                                others.append(c)
                    else:
                        # Fallback if no context exists OR we are using Stanza
                        for c in distractor_opts:
                            c_pos = get_candidate_pos(c, target_noun_context=target_is_noun)
                            is_exact_match = (c_pos == target_pos) or (target_is_noun and c_pos == 'PROPN')
                            
                            if is_exact_match:
                                exact.append(c)
                            elif not target_is_noun and c_pos == 'NOUN':
                                noun_fallback.append(c)
                            elif compatible_pos_list and c_pos in compatible_pos_list:
                                compatible.append(c)
                            elif c_pos not in bad_pos:
                                others.append(c)
                else:
                    for c in distractor_opts:
                        c_pos = get_candidate_pos(c, target_noun_context=target_is_noun)
                        is_exact_match = (c_pos == target_pos) or (target_is_noun and c_pos == 'PROPN')
                        
                        if is_exact_match:
                            exact.append(c)
                        elif not target_is_noun and c_pos == 'NOUN':
                            noun_fallback.append(c)
                        elif compatible_pos_list and c_pos in compatible_pos_list:
                            compatible.append(c)
                        elif c_pos not in bad_pos:
                            others.append(c)
            except Exception as e:
                logging.error(f"Batch POS tagging failed: {e}")
                for c in distractor_opts:
                    c_pos = get_candidate_pos(c, target_noun_context=target_is_noun)
                    if c_pos == target_pos:
                        exact.append(c)
                    elif not target_is_noun and c_pos == 'NOUN':
                        noun_fallback.append(c)
                    elif compatible_pos_list and c_pos in compatible_pos_list:
                        compatible.append(c)
                    elif c_pos not in bad_pos:
                        others.append(c)

            # Priority ordering:
            # Noun targets → exact only (strict)
            # Non-noun targets → exact → compatible → others → noun_fallback (last)
            if target_is_noun:
                distractor_opts = exact
            else:
                distractor_opts = exact + compatible + others + noun_fallback
                if noun_fallback and not exact and not compatible and not others:
                    logging.info(f"POS cascade: using NOUN fallback for non-noun target '{self.words[0]}' (POS={target_pos})")

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

        def candidate_length_ok(candidate, relax_mult=1.0):
            """Check if candidate length matches target within tolerance.
            
            Args:
                candidate: Word to check
                relax_mult: Multiplier for tolerance in relaxed mode (e.g., 3.0 = 3x normal tolerance)
            """
            if not enforce_length_match:
                return True
            sc = strip_punct(candidate)
            if not sc:
                return False
            clen = len(sc)
            # Apply relaxed tolerance multiplier
            effective_tolerance = int(len_tolerance * relax_mult)
            if target_exact_len is not None:
                return abs(clen - target_exact_len) <= effective_tolerance
            if target_preferred_len is not None:
                return abs(clen - target_preferred_len) <= effective_tolerance
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
            # FAST CPU BATCH CACHE: We already tagged them, don't re-run
            if nlp_model is None:
                return False
            key = strip_punct(candidate).lower()
            
            # Dictionary cache lookup (built during batch_tag_words)
            if hasattr(dictionary, 'pos_cache') and key in dictionary.pos_cache:
                return dictionary.pos_cache[key] == 'PROPN'
                
            # Fallback legacy lookup if cache is missing
            if key in _propn_cache:
                return _propn_cache[key]
            
            # DO NOT invoke nlp_sp(clean) one-by-one here. It kills performance.
            # If we don't know it's a PROPN by now, assume it's safe (since we pre-filtered in batch).
            return False
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
            """Pick the most implausible candidate from a pool, respecting filters.
            
            Args:
                relax_length: If True, use 3x tolerance for length matching (still enforces bounds)
                              If False, use strict tolerance from params
            """
            def _find_best(sub_pool):
                local_best = None
                local_best_surp = float('-inf')
                # In relaxed mode, use 3x tolerance but still enforce length bounds
                relax_mult = 3.0 if relax_length else 1.0
                for dist in sub_pool:
                    dist_l = strip_punct(dist).lower()
                    if not dist_l:
                        continue
                    if not candidate_length_ok(dist, relax_mult=relax_mult):
                        continue
                    if dist_l in avoid or dist_l in global_exclude:
                        continue
                    if is_propn_candidate(dist):
                        continue
                    if (not allow_banned) and (dist_l in banned_l):
                        continue
                    if not re.match(r"^[A-Za-zÄÖÜäöüß\u0600-\u06FF]+$", strip_punct(dist)):
                        continue

                    # QUALITY GATE + GRAMMAR GUARD — applied to every fallback stage
                    # (mirrors get_potential_distractors FINAL FILTER so no garbage can
                    # enter through Stage 1/3/5/6 or ultimate desperation either).
                    if lang == 'de':
                        if len(dist_l) < 8:
                            if not re.search(r'[aeiouyäöü]', dist_l):
                                continue
                            if _wordfreq_mod is not None:
                                if _wordfreq_mod.zipf_frequency(dist_l, 'de') < float(params.get('json_min_zipf', 1.5)):
                                    continue
                        if match_casing_only:
                            _pc = getattr(dictionary, 'pos_cache', {})
                            if dist_l in _pc:
                                _is_noun_d = _pc[dist_l] in ('NOUN', 'PROPN')
                                if not target_is_cap and _is_noun_d:
                                    continue
                                if target_is_cap and not _is_noun_d:
                                    continue

                    # NOTE: Similarity filtering is handled by semantic_filter (fastText embeddings)
                    # Levenshtein distance check removed for performance (was causing 6.4B+ operations)
                    
                    s = candidate_min_surprisal(dist)
                    if s is None:
                        continue
                    if s > local_best_surp:
                        local_best_surp = s
                        local_best = dist
                return local_best, local_best_surp

            if target_pos and match_noun_pos:
                exact_pool = [c for c in pool if get_candidate_pos(c, target_noun_context=target_is_noun) == target_pos]
                best_cand, best_surp = _find_best(exact_pool)
                if best_cand is not None:
                    return best_cand, best_surp
                if target_is_noun: 
                    # Nouns MUST be nouns, no fallback allowed
                    return None, float('-inf')
            
            return _find_best(pool)

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
                            if t and (t in dist_l or dist_l in t):
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
                if target_pos and match_noun_pos and not skip_cascade:
                    exact_pool = [c for c in distractor_opts if get_candidate_pos(c, target_noun_context=target_is_noun) == target_pos]
                    best_candidate = _find_best_match(exact_pool)
                    
                    if not best_candidate and not target_is_noun:
                        best_candidate = _find_best_match(distractor_opts)
                else:
                    # Non-cascade path: prioritize non-nouns if target is lowercase
                    if match_noun_pos and not target_is_noun:
                        # Try to find a non-noun in the pool first
                        non_noun_pool = [c for c in distractor_opts if dictionary.pos_cache.get(c.lower()) not in ('NOUN', 'PROPN')]
                        if non_noun_pool:
                            best_candidate = _find_best_match(non_noun_pool)
                    
                    if not best_candidate:
                        best_candidate = _find_best_match(distractor_opts)

                if best_candidate:
                        # Apply grammatical casing based on the distractor's class
                        self.distractor = _normalize_distractor_token(best_candidate, dictionary, lang=lang,
                                                                      target_token=self.words[0] if self.words else "",
                                                                      match_casing_only=match_casing_only)

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
                if t and (t in dist_l or dist_l in t):
                    skip = True
                    break
            if skip:
                continue
            qualified_candidates.append(dist)
        
        # Apply semantic dissimilarity filter if enabled
        if params.get('semantic_filter', False) and qualified_candidates:
            sem_filter = _get_semantic_filter()
            if sem_filter:
                target_word = strip_punct(self.words[0]) if self.words else ""
                pre_filter_count = len(qualified_candidates)
                qualified_candidates = sem_filter.apply_semantic_filter(
                    target_word, qualified_candidates, params
                )
                if len(qualified_candidates) < pre_filter_count:
                    print(f"    [Semantic] Filtered {pre_filter_count} → {len(qualified_candidates)} candidates")
            
        # 1.5 Batch tag candidates for POS persistence (German/Arabic)
        if qualified_candidates and hasattr(dictionary, 'batch_tag_words'):
            dictionary.batch_tag_words(qualified_candidates, params=params)

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
                # Use model_batch_size for scoring, and chunk_size for the batches sent to map back to words
                m_batch_size = getattr(model, 'model_batch_size', int(params.get('model_batch_size', 256)))
                chunk_size = int(params.get('chunk_size', 512))
                
                all_scores = []
                for i in range(0, len(qualified_candidates), chunk_size):
                    chunk = qualified_candidates[i : i + chunk_size]
                    try:
                        chunk_scores = model.get_surprisal_batch_from_hidden(hidden, chunk, batch_size=m_batch_size)
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
            # Fallback stage 1: Use distractor_opts without surprisal threshold
            logging.debug(f"FALLBACK Stage 1: Relaxed pool (no surprisal threshold) for item {self.id}, label {self.lab}")
            fallback = list(distractor_opts)
            random.shuffle(fallback)
            cand, cand_surp = pick_best_from_pool(fallback, allow_banned=False)
            if cand is None and allow_banned_fallback:
                cand, cand_surp = pick_best_from_pool(fallback, allow_banned=True)
            desired_len = target_exact_len if target_exact_len is not None else target_preferred_len
            if cand is None and enforce_length_match and desired_len is not None:
                # Fallback stage 2: Strict exact-length fallback from full in-memory dictionary first.
                logging.debug(f"FALLBACK Stage 2: Full dictionary exact-length search for item {self.id}, label {self.lab}")
                try:
                    # O(1) indexed lookup — uses the pre-built words_by_len index
                    # (keyed by distractor.len at startup) instead of a linear scan
                    # through all ~100k dictionary words.  The old code scanned
                    # getattr(dict, 'words', []) — 'dict' was also the Python
                    # builtin, so the stage always returned [] anyway.
                    exact_full_pool = [w.text for w in
                                       getattr(dictionary, 'words_by_len', {}).get(desired_len, [])]
                except Exception:
                    exact_full_pool = []
                # Enforce noun/non-noun split in casing mode regardless of pos_filter.
                # Previously this check required pos_filter to be truthy, but
                # pos_filter=None in casing-only mode — so the guard was never applied.
                if match_casing_only and exact_full_pool:
                    _is_noun_check = lambda w: (
                        (hasattr(dictionary, 'pos_cache') and dictionary.pos_cache.get(w.lower()) in ('NOUN', 'PROPN'))
                        or (hasattr(dictionary, 'has_titlecase_variant') and dictionary.has_titlecase_variant(w))
                    )
                    if not target_is_cap:
                        exact_full_pool = [w for w in exact_full_pool if not _is_noun_check(w)]
                    else:
                        exact_full_pool = [w for w in exact_full_pool if _is_noun_check(w)]
                if exact_full_pool:
                    random.shuffle(exact_full_pool)
                    cand, cand_surp = pick_best_from_pool(exact_full_pool, allow_banned=False)
                    if cand is None and allow_banned_fallback:
                        cand, cand_surp = pick_best_from_pool(exact_full_pool, allow_banned=True)
            if cand is None:
                # Fallback stage 3: Relax length requirements
                logging.debug(f"FALLBACK Stage 3: Relaxed length matching for item {self.id}, label {self.lab}")
                cand, cand_surp = pick_best_from_pool(fallback, allow_banned=False, relax_length=True)
                if cand is None and allow_banned_fallback:
                    cand, cand_surp = pick_best_from_pool(fallback, allow_banned=True, relax_length=True)
            if cand is None:
                # Fallback stage 5: Emergency pool with POS-aware cascade
                logging.warning(f"FALLBACK Stage 5: Emergency pool (800 random words) for item {self.id}, label {self.lab}")
                try:
                    # --- HYPER-SPEED OPTIMIZATION: USE PRE-INDEXED EMERGENCY POOL ---
                    target_len = target_exact_len or desired_len or 5
                    emergency_pool = dictionary.get_emergency_pool(target_len, is_noun=target_is_noun)
                    
                    if emergency_pool:
                        # We still want to perform a small sample check to find the 'best' surprisal word
                        sample_n = min(800, len(emergency_pool))
                        fallback_emerg = random.sample(emergency_pool, sample_n)
                        
                        cand, cand_surp = pick_best_from_pool(fallback_emerg, allow_banned=False)
                        if cand is None and allow_banned_fallback:
                            cand, cand_surp = pick_best_from_pool(fallback_emerg, allow_banned=True)
                        if cand is None:
                            cand, cand_surp = pick_best_from_pool(fallback_emerg, allow_banned=False, relax_length=True)
                            if cand is None and allow_banned_fallback:
                                cand, cand_surp = pick_best_from_pool(fallback_emerg, allow_banned=True, relax_length=True)
                except Exception as e:
                    logging.error(f"Emergency pool construction failed: {e}")
            
            if cand is None:
                # Fallback stage 6: Last resort - any word from distractor_opts
                logging.warning(f"FALLBACK Stage 6: Desperation mode (any word from pool) for item {self.id}, label {self.lab}")
                fallback_pool = list(distractor_opts)
                if fallback_pool:
                    random.shuffle(fallback_pool)
                    cand, cand_surp = pick_best_from_pool(fallback_pool, allow_banned=False, relax_length=True)
                    if cand is None and allow_banned_fallback:
                        cand, cand_surp = pick_best_from_pool(fallback_pool, allow_banned=True, relax_length=True)

            if cand is None:
                # Final fallback: we exhausted the staged relaxation, leaving it None for desperation blocks
                pass
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
            # THE ULTIMATE DESPERATION FALLBACK:
            # If everything else fails, bypass 'pick_best_from_pool' completely.
            # Directly query the 634k JSON array for the exact length and POS.
            try:
                target_l_len = target_exact_len or 5
                
                if hasattr(dictionary, 'get_emergency_pool'):
                    # Fetch entirely from pre-sorted JSON (lightning fast, guaranteed)
                    if target_is_noun:
                        final_pool = dictionary.get_emergency_pool(target_l_len, is_noun=True)
                    else:
                        final_pool = dictionary.get_emergency_pool(target_l_len, is_noun=False)
                    
                    # If empty, just grab +/- 1 length
                    if not final_pool:
                        if target_is_noun:
                            final_pool = dictionary.get_emergency_pool(target_l_len+1, is_noun=True) + dictionary.get_emergency_pool(target_l_len-1, is_noun=True)
                        else:
                            final_pool = dictionary.get_emergency_pool(target_l_len+1, is_noun=False) + dictionary.get_emergency_pool(target_l_len-1, is_noun=False)

                    if final_pool:
                        random.shuffle(final_pool)
                        for cand in final_pool:
                            cand_l = cand.lower()
                            if cand_l in banned_l or cand_l in avoid:
                                continue
                            # Apply the same guards as _find_best so the ultimate
                            # desperation path cannot produce garbage or casing leaks.
                            if lang == 'de':
                                if len(cand_l) < 8:
                                    if not re.search(r'[aeiouyäöü]', cand_l):
                                        continue
                                    if _wordfreq_mod is not None:
                                        if _wordfreq_mod.zipf_frequency(cand_l, 'de') < float(params.get('json_min_zipf', 1.5)):
                                            continue
                                if is_propn_candidate(cand):
                                    continue
                                if match_casing_only:
                                    _pc2 = getattr(dictionary, 'pos_cache', {})
                                    if cand_l in _pc2:
                                        _is_noun_d2 = _pc2[cand_l] in ('NOUN', 'PROPN')
                                        if not target_is_cap and _is_noun_d2:
                                            continue
                                        if target_is_cap and not _is_noun_d2:
                                            continue
                            best_word = cand
                            break
                        if best_word is None:
                            # Total desperation: ignore banned list, take first clean word
                            for cand in final_pool:
                                cand_l = cand.lower()
                                if lang == 'de' and len(cand_l) < 8:
                                    if not re.search(r'[aeiouyäöü]', cand_l):
                                        continue
                                if is_propn_candidate(cand):
                                    continue
                                best_word = cand
                                break
                        if best_word is None:
                            best_word = final_pool[0]  # absolute last resort
                else:
                    # Legacy fallback for English/etc.
                    emergency_pool = [w.text for w in getattr(dictionary, 'words', []) 
                                    if abs(len(w.text) - (target_exact_len or 5)) <= 2 
                                    and w.text.lower() not in banned_l]
                    if emergency_pool:
                        best_word = random.choice(emergency_pool)
            except Exception as e:
                logging.error(f"Ultimate fallback failed: {e}")
                
            if best_word is None:
                best_word = "wort" # Mathematically impossible with 634k words, but safe.
            best_min_surp = float('-inf')

        # Final casing pass based on DISTRACTOR category
        self.distractor = _normalize_distractor_token(best_word, dictionary, lang=lang,
                                                      target_token=self.words[0] if self.words else "",
                                                      match_casing_only=match_casing_only)

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
        # Stanza/SpaCy POS tagging; else no POS
        nlp_model = _get_nlp_model(lang, params=params)
        if nlp_model is not None:
            for sentence in self.sentences:
                try:
                    if lang == 'de':
                        # Stanza processing. We join words to let Stanza tokenize natively
                        # to avoid pipeline pretokenized conflicts, then map back roughly.
                        # It's better to just pass the string and align.
                        doc = nlp_model(sentence.word_sentence)
                        pos_tags = []
                        
                        # Create character spans for the original words
                        char_pos = 0
                        word_spans = []
                        for w in sentence.words:
                            start = sentence.word_sentence.find(w, char_pos)
                            if start == -1:
                                start = char_pos
                            end = start + len(w)
                            word_spans.append((start, end))
                            char_pos = end

                        # Extract Stanza tokens with character positions
                        stanza_word_tags = []
                        for sent in doc.sentences:
                            for token in sent.tokens:
                                upos_list = [w.upos for w in token.words if w.upos != 'PUNCT']
                                if not upos_list:
                                    upos_list = ['PUNCT']
                                
                                main_pos = upos_list[0]
                                for p in upos_list:
                                    if p in ('NOUN', 'PROPN'):
                                        main_pos = p
                                        break
                                
                                # Use tuple of start_char, end_char, main_pos
                                try:
                                    stanza_word_tags.append((token.start_char, token.end_char, main_pos))
                                except AttributeError:
                                    # Fallback if Stanza version < 1.0 lacks start_char
                                    pass

                        if stanza_word_tags:
                            # Align by character overlap
                            for start, end in word_spans:
                                overlapping_tags = []
                                for t_start, t_end, t_pos in stanza_word_tags:
                                    if t_start >= end or t_end <= start:
                                        continue
                                    if t_pos != 'PUNCT':
                                        overlapping_tags.append(t_pos)
                                
                                if overlapping_tags:
                                    if 'NOUN' in overlapping_tags: final_pos = 'NOUN'
                                    elif 'PROPN' in overlapping_tags: final_pos = 'PROPN'
                                    elif 'VERB' in overlapping_tags: final_pos = 'VERB'
                                    else: final_pos = overlapping_tags[0]
                                else:
                                    final_pos = 'X'
                                pos_tags.append(final_pos)
                        else:
                            # Panic fallback if start_char doesn't exist
                            temp_tags = [word.upos for sent in doc.sentences for word in sent.words]
                            pos_tags = temp_tags[:len(sentence.words)]
                            while len(pos_tags) < len(sentence.words):
                                pos_tags.append('X')

                        sentence.pos_tags = pos_tags
                    else:
                        # SpaCy
                        from spacy.tokens import Doc
                        doc = Doc(nlp_model.vocab, words=sentence.words)
                        for name, proc in nlp_model.pipeline:
                            if name not in ["tok2vec", "tagger", "morphologizer", "attribute_ruler", "lemmatizer"]:
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
                
                # capture context for native tagging
                if not hasattr(self.labels[lab], 'context_words'):
                    self.labels[lab].context_words = sentence.words[:]
                    self.labels[lab].target_idx = i

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
            
            # --- IRONCLAD CASING: Detect target casing here ---
            target_is_capitalized = target[0].isupper() if target else False
            params['target_is_capitalized'] = target_is_capitalized
            
            pos_filter = None
            if params.get('match_noun_pos', False):
                pos_filter = 'NOUN' if target_is_noun_first else '!NOUN'
            
            print(f"  [First] Finding first-word distractor for '{target}' (Cap: {target_is_capitalized})...")
            opts = d.get_potential_distractors(min_length, max_length, min_freq, max_freq, params, pos_filter=pos_filter)
            
            if pos_filter == '!NOUN':
                num_req = int(params.get('num_to_test', 100))
                if len(opts) < (num_req // 2):
                    # We STILL enforce !NOUN even in fallback if match_casing_only is True
                    fallback_filter = '!NOUN' if params.get('match_casing_only', False) else None
                    fallback_opts = d.get_potential_distractors(min_length, max_length, min_freq, max_freq, params, pos_filter=fallback_filter)
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

            # Fallback: search full dictionary emergency pool for closest valid match.
            best = None
            best_diff = 10**9
            try:
                # Check for 634k JSON pool
                if hasattr(d, 'get_emergency_pool'):
                    t_len = target_len or 5
                    # Mostly non-nouns for first word context, or mix both
                    pool = d.get_emergency_pool(t_len, is_noun=False) + d.get_emergency_pool(t_len, is_noun=True)
                    if not pool:
                        pool = d.get_emergency_pool(t_len+1, is_noun=False) + d.get_emergency_pool(t_len-1, is_noun=False)
                else:
                    pool = [w.text for w in getattr(d, "words", [])]
            except Exception:
                pool = []
            if pool:
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
            if best is None and pool:
                best = pool[0] # Desperation mode, ignore filters
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
                    first_dist = _normalize_english_distractor_case(first_dist, is_first_word=True, target_token=sentence.words[0])
                sentence.distractors.append(first_dist)
            for i in range(1, len(sentence.labels)):
                lab = sentence.labels[i]
                # we match distractors to their real words on punctuation
                distractor = copy_punct(sentence.words[i], self.labels[lab].distractor)
                if lang == 'en':
                    distractor = _normalize_english_distractor_case(distractor, is_first_word=False, target_token=sentence.words[i])
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
                    apply_postcase = ('german' in model_loc) or ('german' in dict_cls) or ('benjamin' in hf_name) or ('gerpt2' in hf_name)
                except Exception:
                    apply_postcase = False
            _match_casing_only = bool(params.get('match_casing_only', False))
            if apply_postcase:
                try:
                    for j in range(len(sentence.distractors)):
                        target_tok = sentence.words[j] if j < len(sentence.words) else ""
                        sentence.distractors[j] = _normalize_distractor_token(
                            sentence.distractors[j], d, lang=lang, is_first_word=(j==0),
                            target_token=target_tok, match_casing_only=_match_casing_only)
                except Exception:
                    pass
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
