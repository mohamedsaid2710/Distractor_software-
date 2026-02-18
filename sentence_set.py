import logging
from utils import copy_punct, strip_punct
from limit_repeats import Repeatcounter
import re

# lazy SpaCy pipeline (initialized on first use)
_spacy_nlp = None

def _get_spacy_nlp():
    """Return a loaded spaCy German pipeline if available (de_core_news_sm), else None."""
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            try:
                _spacy_nlp = spacy.load('de_core_news_sm')
            except Exception:
                # try a medium model name if present
                try:
                    _spacy_nlp = spacy.load('de_core_news_md')
                except Exception:
                    _spacy_nlp = None
        except Exception:
            _spacy_nlp = None
    return _spacy_nlp


def _split_punct(token):
    m = re.match(r"^(?P<prefix>[^\wÄÖÜäöüß]*)(?P<body>[\wÄÖÜäöüß'-]+)(?P<suffix>[^\wÄÖÜäöüß]*)$", token, re.UNICODE)
    if not m:
        return '', token, ''
    return m.group('prefix'), m.group('body'), m.group('suffix')


def _normalize_distractor_token(token, dict_obj):
    """Normalize casing for a single distractor token using POS from spaCy and
    dictionary titlecase variant when available.
    """
    # preserve placeholder
    if token == 'x-x-x':
        return token
    prefix, body, suffix = _split_punct(token)
    if not body:
        return token
    # Use spaCy for POS
    upos = None
    nlp_sp = _get_spacy_nlp()
    if nlp_sp is not None:
        try:
            doc = nlp_sp(body)
            if doc and len(doc) > 0:
                upos = doc[0].pos_
        except Exception:
            upos = None
    # if noun or proper noun, prefer dictionary titlecase variant
    if upos in ('NOUN', 'PROPN'):
        try:
            title_var = dict_obj.get_titlecase_variant(body)
        except Exception:
            title_var = None
        if title_var:
            new_body = title_var
        else:
            new_body = body.lower().capitalize()
    else:
        # If POS is unavailable, prefer an explicit Titlecase variant from the
        # dictionary when present. Otherwise, preserve any existing capitalization
        # (so earlier canonicalization isn't undone); finally fall back to lower.
        try:
            title_var = dict_obj.get_titlecase_variant(body)
        except Exception:
            title_var = None
        if title_var:
            new_body = title_var
        else:
            # if body already contains uppercase letters, assume it's intentionally cased
            if any(c.isupper() for c in body):
                new_body = body
            else:
                new_body = body.lower()
    return prefix + new_body + suffix

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
            self.distractors = ["x-x-x"]  # we probably shouldn't hard code this in this way, but whatevs
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

    def choose_distractor(self, model, dict, threshold_func, params, banned):
        """Given a parameters specified in params and stuff
        Find a distractor not on banned (banned=already used in same sentence set)
        That hopefully meets threshold"""
        for surprisal in self.surprisals:  # calculate desired surprisal thresholds
            self.surprisal_targets.append(max(params["min_abs"], surprisal + params["min_delta"]))
        # get us some distractor candidates
        min_length, max_length, min_freq, max_freq = threshold_func(self.words)
        distractor_opts = dict.get_potential_distractors(min_length, max_length, min_freq, max_freq, params)
        avoid=[]
        for word in self.words: # it's awkward if the distractor is the same as the real word
            avoid.append(strip_punct(word).lower())
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
        # initialize
        best_word = "x-x-x"
        best_min_surp = 0
        # New strategy: pick the candidate that maximizes the minimum surprisal
        # across the sentences for this label (i.e., best worst-case surprisal).
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
        # (prefer Titlecase variants) and a small currency whitelist so that
        # words like "euro" are recognized as nouns and capitalized.
        if not target_is_noun:
            try:
                currency_whitelist = set(['euro', 'dollar', 'pound', 'yen'])
                for w in self.words:
                    wlow = w.lower()
                    try:
                        # if dict provides an explicit Titlecase variant, treat as noun
                        if hasattr(dict, 'has_titlecase_variant') and dict.has_titlecase_variant(w):
                            target_is_noun = True
                            break
                    except Exception:
                        pass
                    if wlow in currency_whitelist:
                        target_is_noun = True
                        break
            except Exception:
                pass
        # Optional mode: try to match candidate mean surprisal to the target
        # mean surprisal of the real words. This is more precise than simple
        # thresholding and can be enabled with `params['match_surprisal']`.
        match_surprisal_mode = bool(params.get('match_surprisal', False))
        if match_surprisal_mode and len(self.surprisals) > 0:
            # compute target mean surprisal for the real words in this label
            try:
                target_mean = sum(self.surprisals) / float(len(self.surprisals))
            except Exception:
                target_mean = None
            if target_mean is not None:
                best_diff = float('inf')
                best_candidate = None
                for dist in distractor_opts:
                    dist_l = strip_punct(dist).lower()
                    if dist_l in banned_l or dist_l in avoid:
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
                        best_candidate = dist
                if best_candidate:
                    try:
                        cand = dict.canonical_case(best_candidate)
                    except Exception:
                        cand = best_candidate
                    try:
                        if target_is_noun:
                            try:
                                title_var = dict.get_titlecase_variant(best_candidate)
                            except Exception:
                                title_var = None
                            if title_var:
                                cand = title_var
                            else:
                                cand = cand[0:1].upper() + cand[1:]
                        else:
                            cand = cand.lower()
                    except Exception:
                        pass
                    self.distractor = cand
                    return self.distractor

        # fall back to previous "best worst-case surprisal" strategy
        for dist in distractor_opts:
            dist_l = strip_punct(dist).lower()
            if dist_l in banned_l or dist_l in avoid:
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
            # compute minimum surprisal of this candidate across all sentences
            min_surp_val = float('inf')
            for i in range(len(self.probs)):
                dist_surp = candidate_surprisal(i, dist)
                if dist_surp < min_surp_val:
                    min_surp_val = dist_surp
            if min_surp_val is None:
                continue
            if min_surp_val > best_min_surp:
                best_min_surp = min_surp_val
                best_word = dist
            # if any candidate already meets all surprisal targets, take it immediately
            meets_all = True
            for i in range(len(self.probs)):
                if candidate_surprisal(i, dist) < self.surprisal_targets[i]:
                    meets_all = False
                    break
            if meets_all:
                # apply canonical casing from the dictionary before assigning
                try:
                    cand = dict.canonical_case(dist)
                except Exception:
                    cand = dist
                # if target is a noun, prefer an exact Titlecase variant from the
                # dictionary (falls back to simple capitalization); otherwise
                # return lowercase
                try:
                    if target_is_noun:
                        try:
                            title_var = dict.get_titlecase_variant(dist)
                        except Exception:
                            title_var = None
                        if title_var:
                            cand = title_var
                        else:
                            cand = cand[0:1].upper() + cand[1:]
                    else:
                        cand = cand.lower()
                except Exception:
                    pass
                self.distractor = cand
                return self.distractor
        # apply canonical casing from the dictionary before assigning
        try:
            best_word = dict.canonical_case(best_word)
        except Exception:
            best_word = best_word
        # if target isn't a noun, prefer lowercased fallback
        try:
            if not target_is_noun:
                best_word = best_word.lower()
        except Exception:
            pass
        # if target is noun and a Titlecase variant exists in the static JSON,
        # prefer that exact variant (use dictionary helper if available).
        try:
            if target_is_noun:
                # attempt to fetch a precise titlecase variant from the dict
                title_var = None
                try:
                    title_var = dict.get_titlecase_variant(best_word)
                except Exception:
                    title_var = None
                if title_var:
                    best_word = title_var
                else:
                    # as a fallback, capitalize first letter
                    best_word = best_word[0:1].upper() + best_word[1:]
        except Exception:
            pass
        logging.warning("Could not find a word to meet threshold for item %s, label %s, returning %s with %f min surp instead",
            self.id, self.lab, best_word, best_min_surp)
        self.distractor = best_word
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
            if self.first_labels & self.label_ids != set():
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

    def make_labels(self):
        """Regroups the stuff in the sentence items into by-label groups"""
        # SpaCy-only POS tagging; else no POS
        nlp_sp = _get_spacy_nlp()
        if nlp_sp is not None:
            for sentence in self.sentences:
                try:
                    doc = nlp_sp(sentence.word_sentence)
                    pos_tags = [t.pos_ for t in doc]
                    if len(pos_tags) == len(sentence.words):
                        sentence.pos_tags = pos_tags
                    else:
                        sentence.pos_tags = None
                except Exception:
                    sentence.pos_tags = None
        else:
            for sentence in self.sentences:
                sentence.pos_tags = None
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
        banned = repeats.banned[:] #don't allow duplicate distractors within the set
        for label in self.labels.values(): #get distractors for each label
            dist = label.choose_distractor(model, d, threshold_func, params, banned)
            banned.append(dist)
            repeats.increment(dist)
        for sentence in self.sentences: #give the sentences the distractors
            for i in range(1, len(sentence.labels)):
                lab = sentence.labels[i]
                # we match distractors to their real words on punctuation
                distractor = copy_punct(sentence.words[i], self.labels[lab].distractor)
                sentence.distractors.append(distractor)
            # Ensure the very first distractor slot remains the hard-coded placeholder
            try:
                sentence.distractors[0] = "x-x-x"
            except Exception:
                # if for some reason the list is empty or malformed, ensure placeholder present
                sentence.distractors.insert(0, "x-x-x")
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
            if apply_postcase:
                try:
                    for j in range(len(sentence.distractors)):
                        sentence.distractors[j] = _normalize_distractor_token(sentence.distractors[j], d)
                except Exception:
                    pass
            sentence.distractor_sentence = " ".join(sentence.distractors) #and in sentence_format

    def clean_up(self):
        """Removes memory intensive things like label items and prob distributions"""
        self.labels = {}
        for sentence in self.sentences:
            sentence.probs = []
