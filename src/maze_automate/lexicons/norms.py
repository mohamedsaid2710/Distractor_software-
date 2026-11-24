"""Dominant part of speech per word, from published lexical norms.

Why norms rather than a tagger
------------------------------
The POS lexicon used to be built by running a tagger on each candidate word
in isolation.  Taggers are built for context; on a bare word they guess, and
the guesses were systematically wrong: German HanTa saw every word
capitalised, so *gut*, *habe*, *waren*, *mal* came back NOUN, and a Stanza
proper-noun cross-check on single capitalised words made *von*, *du*, *oder*
PROPN; English spaCy tagged *frequency* VERB and *crib* PROPN; the shipped
Arabic lexicon held ~6,100 suffixed nouns as PRON and ~1,800 valid
accusative forms as X.

A word's POS for Maze purposes is its dominant reading, which is a property
of the word across a corpus.  Each loader below returns, per lowercase word,
``(upos, share)``: the Universal POS tag of the dominant reading and the
proportion of the word's uses that reading accounts for.  The share drives
the ambiguity filter (`pos_min_share` in the params file).

Sources (raw files live in data/norms/, which is gitignored)
------------------------------------------------------------
- English: SUBTLEX-US with PoS (Brysbaert, New, & Keuleers, 2012). CLAWS
  tags counted over 51 million subtitle tokens.  No licence is stated by the
  authors; treated as non-commercial research use.
- German: dlexDB annotated types (Heister et al., 2011), DWDS core corpus of
  the 20th century, STTS tags counted per word form.  Licence CC BY-SA 4.0.
  Word forms are case-sensitive there; counts for "gut" and "Gut" are pooled
  under "gut", which is exactly what settles the ambiguity.
- Arabic: CAMeL Tools (Obeid et al., 2020), calima-msa-r13.  The dominant
  POS is the morphological analyzer's most probable POS when it carries at
  least 60% of the analyzer's prior mass; otherwise the MLE disambiguator's
  choice (learned from the Penn Arabic Treebank) breaks the tie.  The share
  is the analyzer's prior mass for the chosen POS: a model estimate, not a
  corpus count.  Data licence GPL v2.
"""

import collections
import csv
import io
import os
import zipfile

# --- English: SUBTLEX-US (CLAWS-derived labels) -------------------------
SUBTLEX_TO_UPOS = {
    "Noun": "NOUN", "Verb": "VERB", "Adjective": "ADJ", "Adverb": "ADV",
    "Name": "PROPN", "Interjection": "INTJ", "Number": "NUM",
    "Preposition": "ADP", "Pronoun": "PRON", "Determiner": "DET",
    "Article": "DET", "Conjunction": "CCONJ", "To": "PART", "Not": "PART",
    "Ex": "PRON", "Letter": "X",
}


def load_subtlex_us(path):
    """{word: (upos, share)} from the SUBTLEX-US PoS Excel file."""
    import openpyxl  # build-time only: uv run --with openpyxl ...
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = wb.active.iter_rows(values_only=True)
    header = next(rows)
    col = {h: i for i, h in enumerate(header)}
    out = {}
    for r in rows:
        word, dom, share = (r[col["Word"]], r[col["Dom_PoS_SUBTLEX"]],
                            r[col["Percentage_dom_PoS"]])
        upos = SUBTLEX_TO_UPOS.get(dom)
        if word is None or upos is None or share is None:
            continue   # '#N/A' and 'Unclassified' carry no usable tag
        out[str(word).lower()] = (upos, float(share))
    return out


# --- German: dlexDB annotated types (STTS, lowercase in the file) --------
_DLEX_EXTRA = {
    "CARD": "NUM", "PAV": "ADV", "PROAV": "ADV", "PWAV": "ADV",
    "PIATNEG": "DET", "FM": "X", "XY": "X", "TRUNC": "X",
}
# dlexDB's typposlem.tsv has no header.  Verified against the data: column 1
# is the word form, 3 the STTS tag, 8 the absolute annotated-type frequency
# (for a form with two tags, column 8 differs per row and column 14 -- the
# plain type frequency -- is their sum).
_DLEX_FORM, _DLEX_POS, _DLEX_FREQ = 0, 2, 7


def _stts_upos(tag, stts_to_upos):
    t = tag.upper()
    if t.startswith("$"):
        return "PUNCT"
    return stts_to_upos.get(t) or _DLEX_EXTRA.get(t, "X")


def _ss_variants(word):
    """Pre-1996 spellings of a modern 'ss' word: each 'ss' -> 'ß', and all."""
    out = set()
    i = word.find("ss")
    while i != -1:
        out.add(word[:i] + "ß" + word[i + 2:])
        i = word.find("ss", i + 1)
    if "ss" in word:
        out.add(word.replace("ss", "ß"))
    out.discard(word)
    return out


# A share computed from a handful of tokens is meaningless (2 of 2 = 100%).
DLEX_MIN_TOKENS = 10
# dlexDB's corpus is 20th-century German, mostly in the old spelling, while
# the candidate list (wordfreq) is modern: "bloss" has 294 tokens there, all
# tagged as a name, against 13,940 for "bloß" (ADV).  A modern ss-form takes
# the counts of its ß-spelling only when that spelling dominates by this
# factor, so genuinely different words (Masse vs Maße) are not merged.
DLEX_SPELLING_FACTOR = 10


# German long vowels and diphthongs: after these an ß is ß in the old AND the
# new spelling, so dlexDB's (old-spelling) ß is safe to restore.
_LONG_BEFORE_ESZETT = ("ei", "ai", "au", "äu", "eu", "ie")
_VOWELS = set("aeiouyäöü")


def _eszett_is_modern(variant):
    """True if every ß in `variant` is also ß in today's spelling.

    The 1996 reform changed only a ß after a SHORT vowel at the end of a word
    or before a consonant (daß -> dass, muß -> muss, Fluß -> Fluss).  A ß
    before a vowel (Grüße, heißem, ließen) or after a diphthong / ie (weiß,
    heiß) was unaffected.  Any other ß is left as ss: the vowel length
    cannot be read off the spelling, and ss is then the safe choice.
    """
    for i, ch in enumerate(variant):
        if ch != "ß":
            continue
        before_vowel = i + 1 < len(variant) and variant[i + 1] in _VOWELS
        after_long = variant[max(0, i - 2):i] in _LONG_BEFORE_ESZETT
        if not (before_vowel or after_long):
            return False
    return True


def modern_spelling(word, counts):
    """Today's spelling of a wordfreq word, which writes every ß as ss.

    wordfreq lowercases with str.casefold(), which turns ß into ss, so its
    German list has "heissem", "grüsse", "weiss".  The ß-spelling is used
    when dlexDB has it at least DLEX_SPELLING_FACTOR times as often as the
    ss-spelling (so Masse and Maße, two different words, are not merged) and
    every ß in it is one the reform kept (see _eszett_is_modern).
    `counts` maps a lowercase form to its token count.

    When it cannot tell -- dlexDB prefers a ß that the reform may or may not
    have changed -- the word keeps wordfreq's ss.  That is today's spelling
    for dass, muss, Fluss, and the Swiss spelling for gross, Fussball.
    """
    if "ss" not in word:
        return word
    own = counts.get(word, 0)
    dominant = [v for v in _ss_variants(word)
                if counts.get(v, 0) >= DLEX_SPELLING_FACTOR * max(own, 1)]
    if not dominant:
        return word
    sure = [v for v in dominant if _eszett_is_modern(v)]
    if len(sure) != len(dominant):
        return word
    return max(sure, key=lambda v: counts[v])


def load_dlexdb(path, stts_to_upos, keep=None, spellings=None):
    """{word: (upos, share)} from dlexDB's typposlem.tsv (or the zip holding it).

    `keep`, if given, is a set of lowercase words; everything else is skipped
    while streaming, which keeps memory small for a multi-GB file.
    `spellings`, if a dict is given, is filled with {ss-word: ß-word} for every
    kept word whose modern spelling has a ß (see modern_spelling).
    """
    counts = collections.defaultdict(collections.Counter)
    wanted = None
    if keep is not None:
        wanted = set(keep)
        for w in keep:
            wanted |= _ss_variants(w)

    def consume(lines):
        reader = csv.reader(lines, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if len(row) <= _DLEX_FREQ:
                continue
            key = row[_DLEX_FORM].lower()
            if wanted is not None and key not in wanted:
                continue
            try:
                freq = int(row[_DLEX_FREQ])
            except ValueError:
                continue
            if freq > 0:
                counts[key][_stts_upos(row[_DLEX_POS], stts_to_upos)] += freq

    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith("typposlem.tsv"))
            with z.open(name) as raw:
                consume(io.TextIOWrapper(raw, encoding="utf-8", errors="replace"))
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            consume(f)

    if spellings is not None:
        totals = {k: sum(c.values()) for k, c in counts.items()}
        for word in (keep if keep is not None else list(counts)):
            modern = modern_spelling(word, totals)
            if modern != word:
                spellings[word] = modern

    out = {}
    for word in (keep if keep is not None else list(counts)):
        c = collections.Counter(counts.get(word, {}))
        own = sum(c.values())
        for v in _ss_variants(word):
            if sum(counts.get(v, {}).values()) >= DLEX_SPELLING_FACTOR * max(own, 1):
                c.update(counts[v])
        total = sum(c.values())
        if total < DLEX_MIN_TOKENS:
            continue
        tag, n = c.most_common(1)[0]
        out[word] = (tag, n / total)
    return out


# --- Arabic: CAMeL Tools -------------------------------------------------
CAMEL_TO_UPOS = {
    "noun": "NOUN", "noun_quant": "NOUN", "noun_num": "NUM", "noun_prop": "PROPN",
    "adj": "ADJ", "adj_comp": "ADJ", "adj_num": "ADJ",
    "verb": "VERB", "verb_pseudo": "VERB",
    "adv": "ADV", "adv_interrog": "ADV", "adv_rel": "ADV",
    "pron": "PRON", "pron_dem": "PRON", "pron_exclam": "PRON",
    "pron_interrog": "PRON", "pron_rel": "PRON",
    "prep": "ADP", "conj": "CCONJ", "conj_sub": "SCONJ",
    "interj": "INTJ", "digit": "NUM", "punc": "PUNCT",
    "abbrev": "X", "foreign": "X", "latin": "X",
}


def _camel_upos(pos):
    if pos.startswith("part"):
        return "PART"
    return CAMEL_TO_UPOS.get(pos, "X")


def load_camel(words):
    """{word: (upos, share)} for `words`, via CAMeL Tools (calima-msa-r13)."""
    from camel_tools.disambig.mle import MLEDisambiguator  # build-time only
    mle = MLEDisambiguator.pretrained()
    analyzer = mle._analyzer
    table = mle._mle   # word -> best analysis, learned from the Treebank
    out = {}
    for w in words:
        mass = collections.Counter()
        for a in analyzer.analyze(w):
            try:
                mass[_camel_upos(a["pos"])] += 10 ** float(a.get("pos_lex_logprob"))
            except (TypeError, ValueError):
                continue
        total = sum(mass.values())
        if not total and w not in table:
            continue
        # The analyzer's prior decides when it is clear (>= 60% of its mass);
        # the Treebank MLE choice breaks the tie when it is not.  MLE alone
        # inherits news-corpus quirks (بيت -> "Beit", a place-name prefix,
        # tagged PROPN though the analyzer puts 71% on the common noun), and
        # the analyzer alone has no corpus evidence for genuinely split words
        # (كوريا: adjective 37%, noun 37%, proper noun 26%).
        top, top_mass = mass.most_common(1)[0] if mass else (None, 0.0)
        if top is not None and top_mass / total >= 0.6:
            upos = top
        elif w in table:
            upos = _camel_upos(table[w]["pos"])
        else:
            upos = top
        share = (mass[upos] / total) if total else None
        out[w] = (upos, share)
    return out


# Proclitics that make a word an obvious non-word-initial form when shown
# alone as a distractor: the conjunctions w+ / f+ (prc2) and the prepositions /
# particles b+, k+, l+ (prc1).  Farasa's segmenter used to do this at run time.
_CLITIC_PRC2 = ("wa_", "fa_")
_CLITIC_PRC1 = ("bi_", "ka_", "li_", "la_")


def camel_proclitic_words(words):
    """The words whose most likely CAMeL analysis carries w+/f+/b+/k+/l+.

    Uses the MLE disambiguator's choice for the bare word, so a stem that
    merely starts with one of those letters (وزير, لكن) is not affected.
    """
    from camel_tools.disambig.mle import MLEDisambiguator  # build-time only
    mle = MLEDisambiguator.pretrained()
    out = set()
    for w in words:
        try:
            analyses = mle.disambiguate([w])[0].analyses
        except Exception:
            continue
        if not analyses:
            continue
        a = analyses[0].analysis
        if str(a.get("prc2", "0")).startswith(_CLITIC_PRC2) or \
                str(a.get("prc1", "0")).startswith(_CLITIC_PRC1):
            out.add(w)
    return out


# Common Arabic web misspellings: final ه for ة (فتاه), final ى/ي swapped
# (بى, مستوي), a missing or wrong hamza (اعضاء, قرائته), alif tanwin after a
# final hamza (رجاءا).  CAMeL's analyzer is deliberately lenient and accepts
# most of them, so they cannot be caught by analysis.  A word counts as a
# misspelling when one such one-letter change gives a form wordfreq has at
# least ARABIC_SPELLING_FACTOR times as often.
ARABIC_SPELLING_FACTOR = 10
_AR_FINAL_SWAP = {"ه": "ة", "ة": "ه", "ى": "ي", "ي": "ى"}
_AR_HAMZA = "ءأإآئؤ"


def _arabic_spelling_variants(word):
    """Forms one common spelling correction away from `word`."""
    out = set()
    if word[-1] in _AR_FINAL_SWAP:
        out.add(word[:-1] + _AR_FINAL_SWAP[word[-1]])
    if word[0] == "ا":
        out.update(h + word[1:] for h in "أإآ")
    for i, ch in enumerate(word):
        if ch == "آ":
            continue    # آثر, آجل, آخذ are words in their own right, not misspelt أثر
        if ch in _AR_HAMZA:
            out.update(word[:i] + h + word[i + 1:] for h in _AR_HAMZA if h != ch)
        elif ch == "ا" and i > 0:
            # only a hamza on the alif itself: ا -> ؤ would turn راية into رؤية
            out.update(word[:i] + h + word[i + 1:] for h in "أإآ")
    if word.endswith("ءا"):
        out.add(word[:-1])
    out.discard(word)
    return out


def arabic_misspellings(words, freq=None):
    """{misspelled word: its standard spelling} among `words` (wordfreq counts).

    `freq` maps a word to its frequency; it defaults to wordfreq's Arabic
    list (tests pass a dict).
    """
    if freq is None:
        import wordfreq
        def freq(w):
            return wordfreq.word_frequency(w, "ar")
    out = {}
    for w in words:
        if not w:
            continue
        own = freq(w)
        best = max(_arabic_spelling_variants(w), key=freq, default=None)
        if best is not None and freq(best) >= ARABIC_SPELLING_FACTOR * max(own, 1e-12):
            out[w] = best
    return out


SOURCES = {
    "en": {
        "file": "data/norms/en/subtlex_us_pos.xlsx",
        "name": "SUBTLEX-US with PoS",
        "licence": "SUBTLEX-US terms (no licence stated by the authors); non-commercial research use",
        "citation": ("Brysbaert, M., New, B., & Keuleers, E. (2012). Adding "
                     "part-of-speech information to the SUBTLEX-US word "
                     "frequencies. Behavior Research Methods, 44(4), 991-997. "
                     "https://doi.org/10.3758/s13428-012-0190-4"),
        "share": "proportion of the word's subtitle tokens tagged with the dominant PoS",
    },
    "de": {
        "file": "data/norms/de/dlex_v2.zip",
        "name": "dlexDB annotated types (Zenodo, https://doi.org/10.5281/zenodo.15264903)",
        "licence": "CC BY-SA 4.0",
        "citation": ("Heister, J., Würzner, K.-M., Bubenzer, J., Pohl, E., "
                     "Hanneforth, T., Geyken, A., & Kliegl, R. (2011). dlexDB - "
                     "eine lexikalische Datenbank für die psychologische und "
                     "linguistische Forschung. Psychologische Rundschau, 62(1), "
                     "10-20. https://doi.org/10.1026/0033-3042/a000029"),
        "share": "proportion of the word's corpus tokens (all casings) with the dominant tag",
    },
    "ar": {
        "file": None,
        "name": "CAMeL Tools calima-msa-r13 (MLE disambiguator + analyzer)",
        "licence": "GPL-2.0 (CAMeL Tools calima-msa-r13 data)",
        "citation": ("Obeid, O., Zalmout, N., Khalifa, S., Taji, D., Oudah, M., "
                     "Alhafni, B., Inoue, G., Eryani, F., Erdmann, A., & Habash, N. "
                     "(2020). CAMeL Tools: An open source Python toolkit for Arabic "
                     "natural language processing. In Proceedings of the 12th "
                     "Language Resources and Evaluation Conference (pp. 7022-7032). "
                     "European Language Resources Association."),
        "share": "analyzer prior probability mass of the dominant POS (a model estimate)",
    },
}


def load_norms(lang, words, repo_root, stts_to_upos=None, spellings=None):
    """Dispatch to the loader for `lang`; returns {word: (upos, share)}.

    For German, `spellings` (a dict) receives {ss-word: ß-word}; see
    modern_spelling."""
    spec = SOURCES[lang]
    if lang == "ar":
        return load_camel(words)
    path = os.path.join(repo_root, spec["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{spec['name']} not found at {path}; the download commands are "
            f"in the docstring of tools/build_lexicon.py")
    if lang == "en":
        return load_subtlex_us(path)
    return load_dlexdb(path, stts_to_upos, keep=set(words), spellings=spellings)
