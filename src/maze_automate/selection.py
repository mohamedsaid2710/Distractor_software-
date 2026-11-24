"""Selection layer of the Maze pipeline: from candidate pools to final distractors.

Data model (built by `input.py`):
- `Sentence`      — one condition row: words, labels, per-position surprisals and
                    LM contexts ("hiddens" = token-ID prefixes).
- `Label`         — all positions across a `Sentence_Set` that share one label ID;
                    ONE distractor is chosen per label and reused across condition
                    rows (preserves Latin-square comparisons).
- `Sentence_Set`  — all condition rows of one item ID.

The heart is `Label.choose_distractor`, which runs in stages:
 1. thresholds     — per-context surprisal targets (min_abs / min_delta,
                     early-position and short-word boosts);
 2. POS decision   — is the target a noun? build `pos_filter` (noun wall);
 3. pool           — `dictionary.get_potential_distractors` (tiered retrieval);
 4. cheap filters  — length, banned/avoid lists, PROPN, containment;
 5. semantic gate  — optional fastText dissimilarity filter;
 6. batch scoring  — GPU surprisal of every candidate in every context;
 7. selection      — the highest-surprisal candidate that meets every
                     threshold (maximum implausibility);
 8. fallback       — one sequence for every language when nothing passes:
                     best below threshold -> nearest-frequency words ->
                     repeats allowed -> x-x-x placeholder (flagged in the
                     report sidecar's `relaxation` column).

Casing is normalized at the very end by `_normalize_distractor_token`
(German: POS-driven Titlecase via the dictionary's pos_cache; English:
target-mirroring + acronym whitelist; Arabic: no casing).
"""

import logging
from maze_automate import semantic_filter
from maze_automate.utils import (copy_punct, strip_punct, ordered_unique, resolve_repo_path,
                   bare_form, strip_arabic_marks)
import re
import random
import os
import math as _math
try:
    import wordfreq as _wordfreq_mod
except ImportError:
    _wordfreq_mod = None

print("\n" + "="*60)
print("DISTRACTOR MODULE LOADED")
print("="*60 + "\n")

from maze_automate.candidates import (strip_arabic_diacritics,
                                 effective_target_length, MAX_TARGET_LENGTH)


# Common uppercase acronyms to preserve when used as distractors in English.
_EN_ACRONYM_WHITELIST = {
    'ai', 'api', 'bbc', 'cia', 'cpu', 'eu', 'fbi', 'gdp', 'gps', 'gpu',
    'ibm', 'imf', 'ml', 'nasa', 'nato', 'oecd', 'uk', 'un', 'usa', 'uefa',
    'who',
}

_X_PLACEHOLDER_RE = re.compile(r"^x(?:-x)*$", re.IGNORECASE)

# lazy NLP pipeline (initialized on first use)
_nlp_model = {}

class TaggerError(RuntimeError):
    """The input-sentence tagger (spaCy / Stanza) cannot be loaded."""


def _get_nlp_model(lang='de', params=None):
    """Return a loaded NLP pipeline. Stanza for 'de' and 'ar' (input context only), SpaCy for 'en'."""
    global _nlp_model
    lang_lower = str(lang or '').lower()
    
    if lang_lower.startswith('en'):
        key = 'en'
    elif lang_lower.startswith('de'):
        key = 'de'
    elif lang_lower.startswith('ar'):
        key = 'ar'
    else:
        return None
        
    if key in _nlp_model:
        return _nlp_model[key]
        
    if key in ('de', 'ar'):
        # German/Arabic: Stanza tags the input sentences in context.  A missing
        # or unloadable Stanza stops the run: untagged targets would silently
        # switch off pos_match for every position.
        try:
            import stanza
        except ImportError as e:
            raise TaggerError(f"Stanza is not installed ({e}); run `uv sync`") from e
        use_gpu = True
        if params is not None:
            use_gpu = str(params.get('use_gpu', True)).lower() in ('true', '1')
        try:
            try:
                _nlp_model[key] = stanza.Pipeline(key, processors='tokenize,mwt,pos,lemma',
                                                  use_gpu=use_gpu)
            except Exception:
                stanza.download(key)
                _nlp_model[key] = stanza.Pipeline(key, processors='tokenize,mwt,pos,lemma',
                                                  use_gpu=use_gpu)
        except Exception as e:
            raise TaggerError(
                f"could not load the Stanza model for '{key}' ({e}). Run "
                f"`uv run python tools/download_models.py -p <your params file>` "
                f"while online") from e
        return _nlp_model[key]

    elif key == 'en':
        try:
            import spacy
        except ImportError as e:
            raise TaggerError(f"spaCy is not installed ({e}); run `uv sync`") from e
        try:
            _nlp_model[key] = spacy.load('en_core_web_lg')
        except Exception as e:
            raise TaggerError(f"could not load the spaCy model en_core_web_lg ({e}); "
                              f"run `uv sync`") from e
        return _nlp_model[key]


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


# The standard Maze sentence-initial placeholder.  Position 0 has no prior
# context to score against, so there is no real competition there and nothing to
# length-match against; a constant string is also what published Maze materials
# use, which keeps items comparable across studies.
FIRST_WORD_PLACEHOLDER = "x-x-x"


def _placeholder_for_length(length):
    """The first-position placeholder.

    `length` is accepted and ignored.  This used to build one "x" per character
    joined by hyphens, which renders at 2n-1 characters rather than n: a
    10-letter target got "x-x-x-x-x-x-x-x-x-x", 19 characters against a target
    of 10, despite the docstring claiming it matched the target's length.
    """
    return FIRST_WORD_PLACEHOLDER


def _copy_edge_punct_no_case(source_token, token):
    """Copy only leading/trailing punctuation from source token, leave token case untouched."""
    prefix, _body, suffix = _split_punct(source_token)
    return prefix + token + suffix


def _normalize_english_distractor_case(distractor_token, is_first_word=False, target_token=""):
    """English-specific casing normalization based on the distractor's properties and target's case."""
    if _is_x_placeholder_token(distractor_token):
        return distractor_token
    d_prefix, d_body, d_suffix = _split_punct(distractor_token)
    if not d_body:
        return distractor_token

    # Acronyms first.  This check used to sit *below* the target-mirroring
    # branch, which returns for any titlecase target -- so the whitelist was
    # unreachable in the common case and "NASA" was emitted as "Nasa".
    if d_body.lower() in _EN_ACRONYM_WHITELIST:
        return d_prefix + d_body.upper() + d_suffix

    # Interactive Casing: explicitly copy uppercase or titlecase from the target word if available
    if target_token:
        _, t_body, _ = _split_punct(target_token)
        if t_body:
            if t_body.isupper() and len(t_body) > 1:
                return d_prefix + d_body.upper() + d_suffix
            elif t_body.istitle() or (len(t_body) == 1 and t_body.isupper()):
                return d_prefix + d_body.capitalize() + d_suffix

    # Sentence-initial capitalization
    if is_first_word and d_body:
        return d_prefix + d_body[:1].upper() + d_body[1:].lower() + d_suffix

    return distractor_token


def surface_form(token, dict_obj, lang, target_token="", is_first_word=False):
    """The exact string a participant will see -- and therefore the string to score.

    Scoring and emission used to disagree in English: candidates were
    normalized to lowercase before being scored, then mirrored onto the
    target's case at emission.  For a titlecase target, "kitchen" was ranked
    and "Kitchen" delivered -- different BPE token sequences, different
    surprisals -- so the number that drove selection was not the surprisal of
    the string that ends up in the experiment.
    """
    if lang == 'en':
        return _normalize_english_distractor_case(
            token, is_first_word=is_first_word, target_token=target_token)
    return _normalize_distractor_token(
        token, dict_obj, lang=lang, is_first_word=is_first_word,
        target_token=target_token)


def _detect_language(params):
    """Return the language code ('en', 'de' or 'ar') from params.

    set_params() makes `language` required and validates it.  This used to
    fall back to guessing from model_loc / dictionary_class / hf_model_name
    and finally to 'en', while other modules fell back to 'de'.
    """
    from maze_automate.params import LANGUAGES
    lang = (params.get('language') or '').strip().lower()
    if lang not in LANGUAGES:
        raise ValueError(f"params 'language' must be one of {', '.join(LANGUAGES)}, "
                         f"got {params.get('language')!r}")
    return lang


# --- POS matching (`pos_match` in the params file) -------------------------
#   0  off: no POS constraint (the default)
#   1  noun <-> noun and non-noun <-> non-noun
#   2  class <-> class: PROPN counts as NOUN, AUX as VERB, SCONJ as CCONJ
#      (English norms do not separate the two conjunction types), every
#      other Universal POS tag is its own class
# Junk (X) and proper-noun rejection are separate guards and stay on in every
# mode.  This replaced `match_noun_pos` (which promised class matching but was
# a one-directional noun check outside German) and `match_casing_only`.
_POS_CLASS_MERGE = {"PROPN": "NOUN", "AUX": "VERB", "SCONJ": "CCONJ"}


def pos_match_mode(params):
    return int(params.get("pos_match", 0) or 0)


def pos_class(tag, mode):
    """Map a UPOS tag to its matching class under `mode`; None if unknown/off."""
    if not mode or not tag or tag in ("X", "PUNCT"):
        return None
    if mode == 1:
        return "NOUN" if tag in ("NOUN", "PROPN") else "NON-NOUN"
    return _POS_CLASS_MERGE.get(tag, tag)


def pos_mismatch(candidate_tag, target_classes, mode, share=None, min_share=0.0):
    """None if the candidate may fill the slot, else the reason.

    `target_classes` is the set of classes of the target word(s) the
    distractor stands in for (one per condition row; normally one class).
    An empty set means the target's POS is unknown, and nothing is enforced.
    A candidate with no usable tag is rejected whenever matching is on: its
    class cannot be checked, and German casing is derived from the same tag.

    `share` is the proportion of the candidate's uses its dominant POS
    accounts for (from a norms-built lexicon; None if unknown).  Below
    `min_share` (`pos_min_share` in the params file) the word is too
    ambiguous to count as that class, and it is rejected.
    """
    if not mode or not target_classes:
        return None
    cls = pos_class(candidate_tag, mode)
    if cls is None:
        return "pos-unknown"
    if cls not in target_classes:
        return "pos-mismatch"
    if min_share and share is not None and share < min_share:
        return "pos-ambiguous"
    return None


# Closed-class tags: the targets whose frequency nothing else can match.
FUNCTION_WORD_TAGS = {"DET", "PRON", "ADP", "CCONJ", "SCONJ", "AUX", "PART"}


def target_is_function_word(tags, words, dictionary):
    """True if every target word is tagged as a function word.

    The tag is the in-context one (spaCy / Stanza), or the lexicon's when
    that is missing.  Unknown counts as not a function word.
    """
    if not words:
        return False
    for i, word in enumerate(words):
        tag = tags[i] if i < len(tags) else None
        if tag in (None, "X", "PUNCT"):
            tag = lexicon_tag(dictionary, word)
        if tag not in FUNCTION_WORD_TAGS:
            return False
    return True


def lexicon_says_propn(dictionary, word, legacy_cache=None):
    """True if `word` counts as a proper noun for `exclude_propn_candidates`.

    The lexicon answers this, independent of `pos_match`:
      - tagged PROPN -> a name;
      - tagged anything else -> not a name;
      - an English abbreviation (WHO, NASA; see _EN_ACRONYM_WHITELIST) -> a name;
      - no entry in a norms-built lexicon -> a name.  SUBTLEX-US leaves
        names out altogether, so "kevin" or "obama" has no tag rather than
        PROPN; a missing entry is the only sign of a name there.
    """
    key = bare_form(word).lower()
    # English abbreviations are printed and scored in capitals (WHO, NASA)
    # and name an organisation or a place: proper nouns, whatever the
    # lexicon says about the lowercase form ("who" is PRON there).
    if getattr(dictionary, 'LANG', None) == 'en' and key in _EN_ACRONYM_WHITELIST:
        return True
    cache = getattr(dictionary, 'pos_cache', None) or {}
    if key in cache:
        return cache[key] == 'PROPN'
    if getattr(dictionary, 'lexicon_from_norms', False):
        return True
    if legacy_cache and key in legacy_cache:
        return legacy_cache[key]
    return False


# Letters a distractor may consist of, per language.
_SCRIPT_RE = {
    "en": re.compile(r"^[A-Za-z]+$"),
    "de": re.compile(r"^[A-Za-zÄÖÜäöüß]+$"),
    "ar": re.compile(r"^[\u0600-\u06FF]+$"),
}


def load_exclude_words(params):
    """The params file's exclude list (`exclude_words`), lowercased."""
    out = set()
    path = resolve_repo_path(params.get('exclude_words', None))
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                w = line.strip()
                if w and not w.startswith('#'):
                    out.add(w.lower())
    return out


def candidate_problem(candidate, *, lang, dictionary, target_words, length_ok,
                      avoid, exclude, banned, exclude_propn, target_classes,
                      pos_mode, min_share, allow_banned=False):
    """None if `candidate` may be a distractor, else the reason it may not.

    The ONE rule set every position uses -- sentence-initial or not, main
    path or fallback, in every language.  Each language contributes only its
    own letters (_SCRIPT_RE) and its own lexicon.
    """
    base = bare_form(candidate)
    cand_l = base.lower()
    if not cand_l:
        return 'empty'
    script = _SCRIPT_RE.get(lang)
    if script is not None and not script.match(base):
        return 'script'
    if not length_ok(candidate):
        return 'length'
    if cand_l in avoid:
        return 'is-target-or-local-repeat'
    if cand_l in exclude:
        return 'excluded'
    if (not allow_banned) and cand_l in banned:
        return 'banned'
    if exclude_propn and lexicon_says_propn(dictionary, candidate):
        return 'propn'
    # Containment either way is a giveaway ("cat" vs "cats").
    for t in target_words:
        t = bare_form(t).lower()
        if t and (t in cand_l or cand_l in t):
            return 'contains-target'
    tag = getattr(dictionary, 'pos_cache', {}).get(cand_l, None)
    if tag == 'X':
        return 'pos-x'          # junk, in every language
    shares = getattr(dictionary, 'pos_share', {}) or {}
    return pos_mismatch(tag, target_classes, pos_mode,
                        share=shares.get(cand_l), min_share=min_share)


def lexicon_tag(dictionary, word):
    """The POS lexicon's tag for `word` (lowercase bare-form key)."""
    key = bare_form(word).lower()
    return getattr(dictionary, "pos_cache", {}).get(key) if key else None


def target_pos_classes(tags, words, dictionary, mode):
    """Classes of the target word(s), from their in-context tags.

    A position whose context tag is missing or X falls back to the lexicon
    tag of the word itself.
    """
    classes = set()
    for i, word in enumerate(words):
        tag = tags[i] if i < len(tags) else None
        if tag in (None, "X", "PUNCT"):
            tag = lexicon_tag(dictionary, word)
        cls = pos_class(tag, mode)
        if cls is not None:
            classes.add(cls)
    return classes




def _get_german_grammatical_case(token, dict_obj, is_first_word=False, target_token=""):
    """German casing from the distractor's own POS.

    Rules (German orthography):
      - Nouns (NOUN/PROPN) -> Titlecase
      - Everything else     -> lowercase
      - Sentence-initial    -> capitalize first letter regardless of POS

    A word with no lexicon tag is looked up with HanTa (get_titlecase_variant)
    and otherwise lowercased.  That fail-open is why `match_casing_only`
    (mirror the target's casing) once existed; it was removed when
    `pos_match` arrived, because with `pos_match >= 1` an untagged candidate
    is rejected before it can be chosen, so every chosen word has a tag.
    `target_token` is accepted for call compatibility and not used.
    """
    if _is_x_placeholder_token(token):
        return token
    prefix, body, suffix = _split_punct(token)
    if not body:
        return token

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

def _normalize_distractor_token(token, dict_obj, lang='de', is_first_word=False, target_token=""):
    """Normalize casing for a single distractor token by its own grammatical POS."""
    if lang == 'de':
        return _get_german_grammatical_case(token, dict_obj, is_first_word=is_first_word,
                                            target_token=target_token)
    
    # Non-German fallback (e.g. English as before)
    if _is_x_placeholder_token(token):
        return token
    prefix, body, suffix = _split_punct(token)
    if not body:
        return token
    return prefix + body.lower() + suffix

def surprisal_thresholds(words, surprisals, positions, params):
    """(thresholds, is_early, is_short): the surprisal each context must reach.

    threshold = max(min_abs, target surprisal + min_delta)
                + early_position_boost   if any position is 1..early_position_count
                + short_word_boost       if any target has <= short_word_max_len letters
    A short target uses `min_abs_short` instead of `min_abs` when it is set: a
    flat absolute floor is unreachable for function words (English
    `min_abs: 25` asks for p <= 3e-8).

    Position 0 is the sentence-initial placeholder, so "early" starts at 1.
    `positions` are word indices, not label IDs (make_labels renames labels to
    "1_L5", which once made the early boost silently never fire).
    """
    early_boost = float(params.get('early_position_boost', 0))
    early_count = int(params.get('early_position_count', 2))
    is_early = any(p is not None and 1 <= p <= early_count for p in positions)

    short_max = int(params.get('short_word_max_len', 3))
    is_short = any(bare_form(w) and len(bare_form(w)) <= short_max for w in words)

    min_abs = params["min_abs"]
    if is_short and params.get("min_abs_short") is not None:
        min_abs = float(params["min_abs_short"])

    out = []
    for surprisal in surprisals:
        base = max(min_abs, surprisal + params["min_delta"])
        if is_early:
            base += early_boost
        if is_short:
            base += float(params.get('short_word_boost', 0))
        out.append(base)
    return out, is_early, is_short


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
            seen, dups = set(), []
            for lab in labels:
                if lab in seen and lab not in dups:
                    dups.append(lab)
                seen.add(lab)
            raise ValueError(
                f"item {id}: duplicate label(s) {dups} in sentence {' '.join(words)!r}")

    def do_model(self, model):
        """Run the model and record surprisals at each position"""
        hidden = model.empty_sentence()  # initialize
        for i in range(len(self.words) - 1):  # use labels to index
            # Without Arabic diacritics: candidates are scored bare, and the
            # Arabic models are trained on unvowelled text.
            hidden, self.probs[self.labels[i + 1]] = model.update(
                hidden, strip_arabic_marks(self.words[i]))
            # store a shallow copy of the hidden/context after adding this word
            try:
                self.hiddens[self.labels[i + 1]] = list(hidden)
            except Exception:
                # if hidden is not list-like, store as-is
                self.hiddens[self.labels[i + 1]] = hidden

    def do_surprisal(self, model):
        """Surprisal (bits) of each target word, summed over all its subtokens.

        The target is scored exactly like a candidate: the bare word, without
        attached punctuation or Arabic diacritics, in the same left context.  It used to be scored
        with its punctuation ("mat." = " mat" + "."), which added the
        surprisal of the period or comma to the target only and so raised its
        threshold, while candidates were always scored bare.
        """
        for i in range(1, len(self.labels)):  # zeroeth position doesn't count
            lab = self.labels[i]
            word = bare_form(self.words[i]) or self.words[i]
            self.surprisal[lab] = model.get_surprisal_from_hidden(self.hiddens[lab], word)


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
        # Word indices (within their sentence) that this label covers.  Needed
        # because a label ID is not a position: `make_labels` rewrites IDs to
        # "1_L5" form, and user-supplied labels are arbitrary strings anyway.
        self.positions = []
        # How this label's distractor was arrived at, for the report sidecar.
        # Without it a desperation-fallback word is indistinguishable from a
        # clean one in the output file, so there is no way to state how many
        # items actually met the criteria.
        self.report = {}

    def add_sentence(self, word, probs, surprisal, hidden=None, position=None):
        """Given a position that belongs in the label, add it's attributes to our lists"""
        self.words.append(word)
        self.probs.append(probs)
        self.surprisals.append(surprisal)
        self.hiddens.append(hidden)
        self.positions.append(position)
        # pos will be filled by make_labels if available; placeholder None
        self.pos.append(None)

    def choose_distractor(self, model, dictionary, threshold_func, params, banned, local_banned=None):
        """Pick ONE distractor for this label (see module docstring for the stage map).

        Args:
            model:          language-model scorer (surprisal provider).
            dictionary:     candidate-pool provider + POS cache authority.
            threshold_func: maps target words -> (length, frequency) windows.
            params:         full parameter dict from the params file.
            banned:         globally over-used words (Repeatcounter) plus words
                            already chosen for OTHER labels in this set.
            local_banned:   words chosen within the current sentence set only.

        Returns the chosen distractor (already casing-normalized); also stored
        on `self.distractor`.
        """
        if local_banned is None:
            local_banned = []
        # Work on a private copy.  The per-target keys written below
        # (target_is_noun, target_zipf, target_exact_length,
        # target_is_capitalized) used to go into the caller's dict and stay
        # there, so any early return left the NEXT label filtering against the
        # previous label's values -- and target_is_noun is an inviolable
        # noun/non-noun filter in German.
        params = dict(params)
        lang = _detect_language(params)
        pos_mode = pos_match_mode(params)

        # Per-context surprisal thresholds (see surprisal_thresholds).
        self.surprisal_targets, is_early, is_short_word = surprisal_thresholds(
            self.words, self.surprisals, self.positions, params)

        # --- POS decision (`pos_match`) ---
        # The target's class comes from its in-context tag (spaCy / Stanza in
        # make_labels), falling back to the lexicon tag of the word.  This
        # used to be a tangle of per-language heuristics (titlecase variants,
        # a currency whitelist, a lookup that returned None before it ever
        # consulted the lexicon whenever the input tagger was missing).
        target_classes = target_pos_classes(self.pos, self.words, dictionary, pos_mode)
        target_is_noun = "NOUN" in target_classes
        pos_filter = None
        if pos_mode and len(target_classes) == 1:
            # Pre-shapes the dictionary's pool; admissibility() makes the call.
            pos_filter = 'NOUN' if target_is_noun else '!NOUN'
        if pos_mode and not target_classes:
            logging.warning("Item %s label %s: POS of %r unknown; pos_match not "
                            "enforced for this position.", self.id, self.lab,
                            self.words[0] if self.words else "")

        exclude_propn_candidates = bool(params.get('exclude_propn_candidates', False))

        # Ironclad Casing: Inject target casing into params for this word
        orig_target = self.words[0] if self.words else ""
        target_stripped = bare_form(orig_target)
        target_is_cap = target_stripped[0].isupper() if target_stripped else False
        params['target_exact_length'] = len(target_stripped) if target_stripped else None
        # Calculate target Zipf for frequency-matched fallback lookups.
        # This used to hardcode 'de', so English and Arabic targets were looked
        # up in the German frequency table (English "sentence" scored 2.38
        # instead of 4.58) and every frequency-matched fallback aimed at the
        # wrong band.
        target_zipf = 5.0 # default
        if _wordfreq_mod:
            lookup_form = target_stripped.lower()
            if lang == 'ar':
                lookup_form = strip_arabic_diacritics(lookup_form)
            target_zipf = _wordfreq_mod.zipf_frequency(lookup_form, lang)
        params['target_zipf'] = target_zipf

        # NOTE: PRE-TAGGING now handled at SentenceSet-level for optimal batch performance

        import time
        START_TB = time.time()
        print(f"  [Batch] Finding distractors for '{orig_target}' (Cap: {target_is_cap})...")
        params['target_is_noun'] = target_is_noun
        # Short function words get no frequency band (`short_word_freq_match`).
        params['target_is_function_word'] = target_is_function_word(
            self.pos, self.words, dictionary)
        min_length, max_length, min_freq, max_freq = threshold_func(self.words, params)
        distractor_opts = dictionary.get_potential_distractors(min_length, max_length, min_freq, max_freq, params, pos_filter=pos_filter)
        self.report['pool_size'] = len(distractor_opts) if distractor_opts else 0
        self.report['freq_widen_steps'] = getattr(dictionary, 'last_freq_widen_steps', 0)
        self.report['target_zipf'] = round(target_zipf, 3)
        # The band after any widening, so `in_band` judges the distractor
        # against the window that was actually searched.
        _band = getattr(dictionary, 'last_band', None) or (min_freq, max_freq)
        self.report['band_min_zipf'] = round(_band[0] / _math.log(10), 3)
        self.report['band_max_zipf'] = round(_band[1] / _math.log(10), 3)
        # Records the class constraint actually applied (e.g. "NOUN",
        # "NON-NOUN", "VERB"), "unknown" if the target's POS was not found,
        # "" when pos_match is 0.
        self.report['pos_filter'] = ("|".join(sorted(target_classes)) or "unknown") if pos_mode else ""
        
        print(f"  [TIMING] Found {len(distractor_opts) if distractor_opts else 0} opts in {time.time()-START_TB:.2f}s")
        # Skip the POS cascade; it is now deprecated for German.

        enforce_length_match = bool(params.get('enforce_length_match', True))
        len_tolerance = int(params.get('len_tolerance', 0))
        target_lengths = []
        for w in self.words:
            sw = bare_form(w)
            if sw:
                target_lengths.append(len(sw))
        target_exact_len = None
        target_preferred_len = None
        # Clamp to the same range the threshold functions search with, or this
        # test rejects every candidate the pool is able to offer: a 1-2 letter
        # English target got a pool of `min_word_len`-length words and then
        # compared them against its own true length with len_tolerance 0.
        # German does not cap target length (get_thresholds_de passes
        # max_len=None), so capping here would reject every candidate offered
        # for a long compound -- the same failure as short English, mirrored.
        _len_cap = None if lang == 'de' else MAX_TARGET_LENGTH
        if target_lengths:
            unique_lens = sorted(set(target_lengths))
            if len(unique_lens) == 1:
                target_exact_len = effective_target_length(unique_lens[0], params,
                                                           max_len=_len_cap)
                if target_exact_len != unique_lens[0]:
                    logging.info(
                        "Item %s label %s: target length %d is outside the lexicon's "
                        "range; matching on length %d instead.",
                        self.id, self.lab, unique_lens[0], target_exact_len)
            else:
                avg = int(round(sum(target_lengths) / float(len(target_lengths))))
                target_preferred_len = effective_target_length(avg, params,
                                                               max_len=_len_cap)

        def candidate_length_ok(candidate):
            """Candidate length within `len_tolerance` of the target's."""
            if not enforce_length_match:
                return True
            sc = bare_form(candidate)
            if not sc:
                return False
            clen = len(sc)
            # Apply relaxed tolerance multiplier
            effective_tolerance = len_tolerance
            if target_exact_len is not None:
                return abs(clen - target_exact_len) <= effective_tolerance
            if target_preferred_len is not None:
                return abs(clen - target_preferred_len) <= effective_tolerance
            return True
        avoid=[]
        for word in self.words: # it's awkward if the distractor is the same as the real word
            avoid.append(bare_form(word).lower())
        for l_banned in local_banned:
            avoid.append(bare_form(l_banned).lower())
            
        global_exclude = load_exclude_words(params)

        # normalize banned list to lowercase for case-insensitive comparison
        banned_l = [b.lower() for b in banned]

        pos_min_share = float(params.get('pos_min_share', 0.0) or 0.0)

        def admissibility(candidate, allow_banned=False):
            """See candidate_problem: the one rule set for every position."""
            return candidate_problem(
                candidate, lang=lang, dictionary=dictionary, target_words=self.words,
                length_ok=candidate_length_ok, avoid=avoid, exclude=global_exclude,
                banned=banned_l, exclude_propn=exclude_propn_candidates,
                target_classes=target_classes, pos_mode=pos_mode,
                min_share=pos_min_share, allow_banned=allow_banned)

        def is_admissible(candidate, **kw):
            return admissibility(candidate, **kw) is None

        semantic_on = bool(params.get('semantic_filter', False))
        target_word = bare_form(self.words[0]) if self.words else ""

        def usable(pool, allow_banned=False):
            """Admissible words of `pool` that also pass the semantic filter."""
            out = [c for c in ordered_unique(pool) if is_admissible(c, allow_banned=allow_banned)]
            if semantic_on and out:
                before = len(out)
                # Compared as shown: fastText vectors are case-sensitive.
                shown = {c: surface_form(c, dictionary, lang, target_token=target_tok)
                         for c in out}
                out = semantic_filter.apply_semantic_filter(target_word, out, params,
                                                            shown=shown)
                if len(out) < before:
                    print(f"    [Semantic] Filtered {before} -> {len(out)} candidates")
            return out

        m_batch_size = getattr(model, 'model_batch_size', int(params.get('model_batch_size', 256)))
        chunk_size = int(params.get('chunk_size', 512))
        target_tok = self.words[0] if self.words else ""

        def score(cands):
            """[(candidate, lowest surprisal across contexts, meets every threshold)].

            Every candidate is scored as the string a participant will see
            (surface_form), in every condition's context; the lowest surprisal
            is the one that counts.
            """
            if not cands:
                return []
            per_context = []
            for j in range(len(self.probs)):
                hidden = self.hiddens[j]
                scores = []
                for i in range(0, len(cands), chunk_size):
                    chunk = [surface_form(c, dictionary, lang, target_token=target_tok)
                             for c in cands[i:i + chunk_size]]
                    try:
                        scores.extend(model.get_surprisal_batch_from_hidden(
                            hidden, chunk, batch_size=m_batch_size))
                    except Exception as e:
                        logging.error(f"Batch scoring failed: {e}")
                        # Slower one-word-at-a-time scoring, same exact sum.
                        scores.extend(model.get_surprisal_from_hidden(hidden, c) for c in chunk)
                per_context.append(scores)
            out = []
            for k, c in enumerate(cands):
                vals = [per_context[j][k] for j in range(len(self.probs))]
                meets = all(v >= t for v, t in zip(vals, self.surprisal_targets))
                out.append((c, min(vals), meets))
            return out

        # --- Main path: score the admissible pool, keep the passing words ---
        best_word, best_min_surp = None, float('-inf')
        qualified = usable(distractor_opts)
        if qualified and hasattr(dictionary, 'batch_tag_words'):
            dictionary.batch_tag_words(qualified, params=params)   # tagger fallback only
        print(f"  [Batch] Scoring {len(qualified)} candidates for target '{target_tok}'...")
        scored = score(qualified)
        passed = sorted([(c, s_) for c, s_, ok in scored if ok],
                        key=lambda x: x[1], reverse=True)

        # Maximum implausibility: the highest-surprisal passing candidate.
        # Short words draw from the top 150 instead, so the same few
        # function-word distractors do not repeat in every item.
        _desired_len = target_exact_len if target_exact_len is not None else target_preferred_len
        _is_short = _desired_len is not None and _desired_len <= 4
        if passed:
            idx = random.randint(0, min(150, len(passed)) - 1) if _is_short else 0
            best_word, best_min_surp = passed[idx]

        # --- Fallback: one sequence, the same for every language ---
        # Each step keeps every rule (length, POS, exclusions, semantic
        # filter) and takes the HIGHEST-surprisal word it finds; the report's
        # `relaxation` column names the step.
        #   below-threshold   the best of the scored pool, though under threshold
        #   nearest-frequency nothing usable in the pool: words of the same
        #                     length nearest in frequency, then the emergency pool
        #   repeats-allowed   as above, but words used elsewhere may repeat
        #                     (allow_banned_fallback, or a short target)
        #   none-admissible   nothing at all: the x-x-x placeholder
        fallback_relaxation = ''
        if best_word is None:
            def best_of(pool_scored):
                return max(pool_scored, key=lambda x: x[1]) if pool_scored else None

            pick, step = best_of(scored), 'below-threshold'

            if pick is None:
                target_len = target_exact_len or _desired_len or 5
                nearby = list(dictionary.get_best_frequency_pool(
                    target_len, target_zipf, n=int(params.get('num_to_test', 200))))
                if pos_filter:
                    nearby += dictionary.get_emergency_pool(target_len, is_noun=target_is_noun)
                else:
                    nearby += (dictionary.get_emergency_pool(target_len, is_noun=True)
                               + dictionary.get_emergency_pool(target_len, is_noun=False))
                extra = [c for c in usable(nearby) if c not in set(qualified)]
                pick, step = best_of(score(extra)), 'nearest-frequency'

                _repeat_exempt = int(params.get("repeat_exempt_max_len", 4))
                allow_repeats = bool(params.get("allow_banned_fallback", False)) or \
                    (_desired_len is not None and _desired_len <= _repeat_exempt)
                if pick is None and allow_repeats:
                    pick, step = best_of(score(usable(list(distractor_opts) + nearby,
                                                      allow_banned=True))), 'repeats-allowed'

            if pick is not None:
                best_word, best_min_surp = pick[0], pick[1]
                fallback_relaxation = step
            else:
                best_word = _placeholder_for_length(0)
                fallback_relaxation = 'none-admissible'
                logging.error(
                    "Item %s label %s: no admissible candidate; emitting the "
                    "placeholder. This position needs manual attention (see the "
                    "report sidecar).", self.id, self.lab)
        used_fallback = bool(fallback_relaxation)

        # Final casing pass based on DISTRACTOR category
        self.distractor = surface_form(best_word, dictionary, lang,
                                       target_token=self.words[0] if self.words else "")

        self.report['used_fallback'] = used_fallback
        self.report['relaxation'] = fallback_relaxation
        self.report['achieved_surprisal'] = (
            None if best_min_surp in (float('-inf'), float('inf'))
            else round(best_min_surp, 3))
        self.report['threshold'] = (round(max(self.surprisal_targets), 3)
                                    if self.surprisal_targets else None)
        self.report['target_surprisal'] = (round(max(self.surprisals), 3)
                                           if self.surprisals else None)
        self.report['is_placeholder'] = _is_x_placeholder_token(best_word)

        # This used to fire on every call, successes included, so the
        # log gave no signal about which items actually missed their threshold.
        if used_fallback:
            logging.warning(
                "Item %s label %s: no candidate met the surprisal threshold; "
                "returning %s from the fallback cascade (min surp %s)",
                self.id, self.lab, self.distractor,
                "n/a" if best_min_surp == float('-inf') else f"{best_min_surp:.2f}")
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
                clash = sorted(str(x) for x in self.first_labels & self.label_ids)
                raise ValueError(
                    f"item {self.id}: label(s) {clash} are used for the first word "
                    f"in one sentence and for a later word in another; the first "
                    f"word's label must not recur later in the same item")
        else:
            raise ValueError(
                f"sentence with id {sentence.id!r} added to item {self.id!r}")

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
                    if lang in ('de', 'ar'):
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
        # One distractor is reused for every position sharing a label, so all those
        # positions must have the SAME target length for length matching to hold.
        # If condition rows put different-length words on one label, we split the
        # label by length ("3" -> "3_L5", "3_L7"), each getting its own distractor.
        new_label_ids = []
        seen_labs = set()
        for sentence in self.sentences:
            for i in range(1, len(sentence.labels)):
                orig_lab = sentence.labels[i]
                target_len = len(bare_form(sentence.words[i]))
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

                self.labels[lab].add_sentence(sentence.words[i], sentence.probs[lab], sentence.surprisal[lab], hidden, position=i)
                # store pos into the last appended slot
                try:
                    self.labels[lab].pos[-1] = pos
                except Exception:
                    pass

    def do_distractors(self, model, d, threshold_func, params, repeats):
        """Choose one distractor per label, then assemble per-sentence rows.

        Order of operations:
        1. (de/ar) hybrid batch-tag the input words to grow the POS cache;
        2. `choose_distractor` per label — each pick is immediately added to the
           banned lists so later labels in this set cannot repeat it;
        3. per sentence: position 0 gets an x-x-x placeholder (or a real word if
           `first_token_placeholder` is False), later positions copy their
           label's distractor with the target's punctuation.  Casing is
           settled when each distractor is chosen (surface_form).
        """
        lang = _detect_language(params)
        
        # --- HYBRID APPROACH: INPUT-LEVEL BATCH TAGGING ---
        # Aggregate all target words and their Stanza POS tags, 
        # then verify them against HanTa (German fallback only).
        if lang in ('de', 'ar') and hasattr(d, 'batch_tag_inputs'):
            input_stanza_tags = {}
            for sent in self.sentences:
                for i, w in enumerate(sent.words):
                    t = strip_punct(w).lower()
                    if t:
                        if t not in input_stanza_tags:
                            input_stanza_tags[t] = set()
                        if hasattr(sent, 'pos_tags') and sent.pos_tags and i < len(sent.pos_tags):
                            pos = sent.pos_tags[i]
                            if pos and pos != 'X':
                                input_stanza_tags[t].add(pos)
            
            if input_stanza_tags:
                logging.info(f"[PERFORMANCE] Hybrid Batch-tagging {len(input_stanza_tags)} target types.")
                d.batch_tag_inputs(input_stanza_tags, params=params)

        banned = repeats.banned[:] #don't allow duplicate distractors within the set
        local_banned = []
        for label in self.labels.values(): #get distractors for each label
            dist = label.choose_distractor(model, d, threshold_func, params, banned, local_banned=local_banned)
            banned.append(dist)
            local_banned.append(dist)
            repeats.increment(dist)

        # One first-word distractor per distinct sentence-initial target, so
        # condition rows of the same item that start with the same word get the
        # same distractor.  Previously every row drew independently.
        first_word_choices = {}
        outer_params = params
        pos_mode = pos_match_mode(params)

        exclude_words = load_exclude_words(params)

        def choose_first_word(sentence):
            """A real-word distractor for position 0 (`first_token_placeholder: False`).

            Same rules as every other position (candidate_problem) and the same
            fallback steps.  The one difference: a sentence-initial word has no
            left context, so it cannot be scored for surprisal, and the choice
            among admissible words is random (seeded).
            """
            params_ = dict(outer_params)
            target = bare_form(sentence.words[0])
            target_len = len(target)
            first_tags = sentence.pos_tags[:1] if getattr(sentence, 'pos_tags', None) else []
            classes = target_pos_classes(first_tags, sentence.words[:1], d, pos_mode)
            is_noun = "NOUN" in classes
            pos_filter = ('NOUN' if is_noun else '!NOUN') if (pos_mode and len(classes) == 1) else None
            params_['target_is_noun'] = is_noun
            params_['target_is_function_word'] = target_is_function_word(
                first_tags, sentence.words[:1], d)
            params_['target_zipf'] = (_wordfreq_mod.zipf_frequency(
                strip_arabic_diacritics(target.lower()) if lang == 'ar' else target.lower(), lang)
                if _wordfreq_mod else 5.0)
            min_l, max_l, min_f, max_f = threshold_func([sentence.words[0]], params_)

            len_tol = int(params_.get('len_tolerance', 0))
            enforce_len = bool(params_.get('enforce_length_match', True))
            want_len = effective_target_length(target_len, params_,
                                               max_len=None if lang == 'de' else MAX_TARGET_LENGTH)

            def length_ok(c):
                return (not enforce_len) or abs(len(bare_form(c)) - want_len) <= len_tol

            avoid = {target.lower()} | {bare_form(b).lower() for b in local_banned}
            banned_l = {bare_form(b).lower() for b in banned}
            min_share = float(params_.get('pos_min_share', 0.0) or 0.0)

            def usable(pool, allow_banned=False):
                return [c for c in ordered_unique(pool) if candidate_problem(
                    c, lang=lang, dictionary=d, target_words=sentence.words[:1],
                    length_ok=length_ok, avoid=avoid, exclude=exclude_words,
                    banned=banned_l, exclude_propn=bool(params_.get('exclude_propn_candidates', False)),
                    target_classes=classes, pos_mode=pos_mode, min_share=min_share,
                    allow_banned=allow_banned) is None]

            pool = d.get_potential_distractors(min_l, max_l, min_f, max_f, params_,
                                               pos_filter=pos_filter)
            nearby = list(d.get_best_frequency_pool(
                want_len, params_['target_zipf'], n=int(params_.get('num_to_test', 200))))
            if pos_filter:
                nearby += d.get_emergency_pool(want_len, is_noun=is_noun)
            else:
                nearby += (d.get_emergency_pool(want_len, is_noun=True)
                           + d.get_emergency_pool(want_len, is_noun=False))

            _repeat_exempt = int(params_.get("repeat_exempt_max_len", 4))
            allow_repeats = bool(params_.get("allow_banned_fallback", False)) or \
                target_len <= _repeat_exempt
            for candidates in (usable(pool), usable(nearby),
                               usable(pool + nearby, allow_banned=True) if allow_repeats else []):
                if candidates:
                    pick = random.choice(candidates)
                    return surface_form(pick, d, lang, target_token=sentence.words[0],
                                        is_first_word=True)
            logging.error("Item %s: no admissible first-word distractor; emitting "
                          "the placeholder.", self.id)
            return _placeholder_for_length(0)

        use_first_placeholder = bool(params.get("first_token_placeholder", True))
        for sentence in self.sentences: #give the sentences the distractors
            sentence.distractors = []
            if use_first_placeholder:
                first_len = len(strip_punct(sentence.words[0]))
                first_placeholder = _placeholder_for_length(first_len)
                sentence.distractors.append(_copy_edge_punct_no_case(sentence.words[0], first_placeholder))
            else:
                # Reuse the choice for an identical sentence-initial target, and
                # register it globally.  Previously every condition row drew its
                # own position-0 distractor with a fresh shuffle, and the result
                # never reached `banned`, `local_banned` or `repeats`, so rows of
                # one item disagreed at position 0 and `max_repeat` was simply
                # not enforced there.
                first_key = strip_punct(sentence.words[0]).lower()
                if first_key in first_word_choices:
                    first_dist = first_word_choices[first_key]
                else:
                    first_dist = choose_first_word(sentence)
                    first_word_choices[first_key] = first_dist
                    banned.append(first_dist)
                    local_banned.append(first_dist)
                    repeats.increment(first_dist)
                sentence.distractors.append(copy_punct(sentence.words[0], first_dist))
            for i in range(1, len(sentence.labels)):
                lab = sentence.labels[i]
                # Casing was settled when the distractor was chosen
                # (surface_form); only the target's punctuation is copied.
                sentence.distractors.append(
                    copy_punct(sentence.words[i], self.labels[lab].distractor))
            sentence.distractor_sentence = " ".join(sentence.distractors) #and in sentence_format

    def clean_up(self):
        """Removes memory intensive things like label items and prob distributions"""
        self.labels = {}
        for sentence in self.sentences:
            sentence.probs = {}


# --- Repeat limits across the whole run (formerly selection.py) ---

class Repeatcounter:
    """Keeps track of how many times each distractor has been used so far in the entire set of all items.
    Provides a list of those that are now banned b/c they have been used too much, and counts the rest.
    Disallows more than max number. If max is 0, no max is enforced."""

    def __init__(self, max_count, exempt_max_len=4):
        """initializes counter.

        `exempt_max_len` is the length at or below which a word is exempt from
        `max_repeat`.  This used to be a hardcoded `len(word_key) > 4` with no
        way to see or change it, so `max_repeat: 1` was silently unenforced for
        every word of four characters or fewer -- which is most distractors for
        short function-word targets.  Set it to 0 to enforce the limit at all
        lengths.
        """
        self.max = max_count
        if max_count == 0:
            self.limit = False
        else:
            self.limit = True
        self.exempt_max_len = int(exempt_max_len)
        self.distractors = dict()
        self.banned = []

    def increment(self, word):
        """adds a new repeat of word to the list, if this puts it up to max, adds it to banned"""
        word_key = word.lower()
        if word_key in self.distractors.keys():
            self.distractors[word_key] += 1
        else:
            self.distractors[word_key] = 1

        if self.limit and len(word_key) > self.exempt_max_len:
            if self.distractors[word_key] >= self.max:
                self.banned.append(word_key)
