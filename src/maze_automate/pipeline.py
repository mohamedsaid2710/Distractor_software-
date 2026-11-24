import logging
import importlib
import sys
import random
from maze_automate import selection, semantic_filter
from maze_automate.params import set_params, ParamsError, LANGUAGES
from maze_automate.selection import Repeatcounter
from maze_automate.input import read_input
from maze_automate.output import (append_results, init_report, append_report, summarize_report)


DEFAULT_SEED = 12345


class LexiconError(RuntimeError):
    """The POS lexicon is missing or unusable and the tagger fallback was declined."""


def confirm_lexicon(dict_class, params, allow_tagger_fallback=None, ask=input):
    """Stop, or ask, when the POS lexicon for this language cannot be used.

    Without a norms-built lexicon every candidate is POS-tagged word by word
    by a tagger, which is unreliable on bare words (HanTa called *gut*,
    *habe*, *waren* nouns).  That used to happen silently.  Now:
    allow_tagger_fallback=True continues with a warning, False stops, and None
    asks on an interactive terminal and stops otherwise (a cluster job must not
    hang on a prompt, and must not fall back unseen either).
    """
    path = getattr(dict_class, "LEXICON_PATH", None)
    if path is None:
        return
    from maze_automate.candidates import lexicon_status
    status = lexicon_status(path)
    if status == "norms":
        return
    lang = params["language"]
    tagger = getattr(dict_class, "FALLBACK_TAGGER", "a tagger")
    what = {"missing": "is missing",
            "unreadable": "cannot be read",
            "tagger-built": "was built by a tagger, not from lexical norms"}[status]
    if tagger is None:
        # No bare-word tagger exists for this language (Arabic): nothing to fall back to.
        raise LexiconError(f"POS lexicon {path} {what}, and {lang} has no fallback "
                           f"tagger.\nBuild it with:  uv run python tools/build_lexicon.py "
                           f"--lang {lang}")
    explain = (f"POS lexicon {path} {what}.\n"
               f"Candidates would be POS-tagged word by word by {tagger}, which is "
               f"unreliable on single words.\n"
               f"Build it with:  uv run python tools/build_lexicon.py --lang {lang}")
    if allow_tagger_fallback is None:
        if sys.stdin is not None and sys.stdin.isatty():
            print("\n" + explain, flush=True)
            reply = ask(f"Continue with {tagger} anyway? [y/N] ").strip().lower()
            allow_tagger_fallback = reply in ("y", "yes")
        else:
            raise LexiconError(explain + "\n(Non-interactive run: pass "
                               "--allow-tagger-fallback to continue with the tagger.)")
    if not allow_tagger_fallback:
        raise LexiconError(explain)
    logging.warning("Continuing WITHOUT a norms lexicon: %s tags candidates word by "
                    "word (%s %s).", tagger, path, what)


# Which dictionary, language model and length/frequency function each
# language uses.  A params file may override any of them with the
# `*_loc` / `*_class` / `*_name` keys (the tests plug in a fake model this way).
LANGUAGE_WIRING = {
    lang: {
        "dictionary": ("maze_automate.candidates", f"wordfreq_{name}_zipf_dict"),
        "model": (f"maze_automate.scoring.{name.lower()}", f"{name}Scorer"),
        "thresholds": ("maze_automate.candidates", f"get_thresholds_{lang}"),
    }
    for lang, name in LANGUAGES.items()
}


def check_language_consistency(params, params_file, dict_class, model_class):
    """Stop if `language` disagrees with the dictionary or model class.

    New languages are added by copying a params file; a half-edited copy
    (language: "de" with the English dictionary) would otherwise run.  The
    classes declare their own language (`LANG` on dictionaries, `LANG_NAME`
    on scorers); a class that declares none is not checked.
    """
    lang = params["language"]
    expected = LANGUAGES[lang]
    dict_lang = getattr(dict_class, "LANG", None)
    if dict_lang is not None and dict_lang != lang:
        raise ParamsError(
            f"{params_file}: language is '{lang}' but dictionary_class "
            f"'{dict_class.__name__}' is for '{dict_lang}'")
    model_lang = getattr(model_class, "LANG_NAME", None)
    if model_lang is not None and model_lang != expected:
        raise ParamsError(
            f"{params_file}: language is '{lang}' ({expected}) but model_class "
            f"'{model_class.__name__}' is for {model_lang}")


def load_from_params(params, params_file, loc_key, default_loc, name_key, default_name):
    """Import params[loc_key] and return its attribute params[name_key].

    The params file wires a language together by naming modules and classes,
    so a typo there used to surface as a bare ModuleNotFoundError or
    AttributeError traceback that never said which params key was wrong.
    """
    loc = params.get(loc_key, default_loc)
    name = params.get(name_key, default_name)
    try:
        module = importlib.import_module(loc)
    except ModuleNotFoundError as e:
        if e.name is not None and (loc == e.name or loc.startswith(e.name + ".")):
            raise ParamsError(
                f"{params_file}: {loc_key} '{loc}' is not a module in this repo") from e
        # The module exists but one of its own imports is not installed.
        raise ImportError(
            f"{loc_key} '{loc}' needs the package '{e.name}', which is not "
            f"installed; run `uv sync` to install all dependencies") from e
    try:
        return getattr(module, name)
    except AttributeError as e:
        raise ParamsError(
            f"{params_file}: {name_key} '{name}' not found in module '{loc}'") from e


def seed_everything(seed):
    """Pin every RNG the pipeline can reach.

    Selection is driven by random.shuffle/randint/sample/choice in
    selection.py and candidates.py, so without this two runs on
    the same input produce different distractors and no before/after
    comparison means anything.  numpy and torch are seeded too because the
    scorers use them; both imports are optional so this stays usable in the
    no-model unit tests.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


#these are the default values, but can be overridden by parameters file 
def run_stuff(infile, outfile, parameters, outformat="delim",
              allow_tagger_fallback=None):
    """Takes an input file, and an output file location
    Does the whole distractor thing (according to specified parameters)
    Writes in outformat.

    Returns the list of item IDs that failed and are therefore missing from
    the output file (empty on a fully successful run)."""
    if outformat not in ["delim", "ibex"]:
        raise ValueError(f"output format must be 'delim' or 'ibex', got {outformat!r}")
    params = set_params(parameters)

    # Reproducibility: pin the RNGs before anything draws from them.
    # `seed: null` in the params file opts out and restores the old
    # nondeterministic behaviour.
    seed = params.get("seed", DEFAULT_SEED)
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError) as e:
            raise ParamsError(
                f"{parameters}: seed must be an integer or null, got {seed!r}") from e
        seed_everything(seed)
        print(f">>> Random seed: {int(seed)}")
    else:
        print(">>> Random seed: unset (output will not be reproducible)")

    # Resolve every name the params file gives before loading anything heavy,
    # so a typo fails in seconds rather than after the dictionary has loaded.
    wiring = LANGUAGE_WIRING[params["language"]]
    dict_class = load_from_params(params, parameters,
                                  "dictionary_loc", wiring["dictionary"][0],
                                  "dictionary_class", wiring["dictionary"][1])
    model_class = load_from_params(params, parameters,
                                   "model_loc", wiring["model"][0],
                                   "model_class", wiring["model"][1])
    threshold_func = load_from_params(params, parameters,
                                      "threshold_loc", wiring["thresholds"][0],
                                      "threshold_name", wiring["thresholds"][1])
    check_language_consistency(params, parameters, dict_class, model_class)
    confirm_lexicon(dict_class, params, allow_tagger_fallback)
    # Fail now, not at the first item, if the filter is on but has no vectors.
    semantic_filter.require_vectors(params)
    # Likewise the input-sentence tagger (spaCy / Stanza): without it every
    # target would be untagged and pos_match silently off.
    selection._get_nlp_model(params["language"], params=params)
    d = dict_class(params)
    sents = read_input(infile)
    m = model_class(params)
    repeats = Repeatcounter(params.get("max_repeat", 0),
                            exempt_max_len=params.get("repeat_exempt_max_len", 4))
    
    
    # PRE-CLEAR OUTFILE to allow incremental appending
    with open(outfile, 'w', encoding='utf-8'):
        pass

    write_report = bool(params.get("write_report", True))
    report_lang = params["language"]
    if write_report:
        init_report(outfile)

    total = len(sents)
    executed_ids = []
    failed_ids = []
    for i, (ss_id, ss) in enumerate(sents.items(), 1):
        # Get tag from the first sentence in the set (most sets only have one)
        tag = ss.sentences[0].tag if ss.sentences else "None"
        print(f"\n>>> [{i}/{total}] Processing Item ID: {ss_id} (Tag: {tag})", flush=True)
        
        try:
            ss.do_model(m)
            ss.do_surprisals(m)
            ss.make_labels(params)
            ss.do_distractors(m, d, threshold_func, params, repeats)
            
            # INCREMENTAL SAVE: Save this sentence set immediately
            append_results(outfile, ss, outformat)
            if write_report:
                append_report(outfile, ss, report_lang,
                              len_tolerance=int(params.get('len_tolerance', 0)))
            executed_ids.append(ss_id)
            print(f"    ✅ Success: Item {ss_id} saved to {outfile}")
            
        except Exception as e:
            print(f"    ❌ ERROR on Item {ss_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_ids.append(ss_id)
        finally:
            # `continue` used to jump over this, so every failed item leaked its
            # full-vocabulary probability tensors for the rest of the run --
            # exactly when memory pressure was already the likely cause.
            ss.clean_up()
    
    if write_report:
        summarize_report(outfile)
    print(f"\n>>> GENERATION COMPLETE. Processed {len(executed_ids)}/{total} items.")
    print(f">>> Successfully saved IDs: {executed_ids}")
    if failed_ids:
        # A run that silently exits 0 having dropped items is indistinguishable
        # from a clean one, which is not something to discover after analysis.
        print(f">>> FAILED IDs ({len(failed_ids)}): {failed_ids}")
        logging.error("%d of %d items failed and are missing from %s: %s",
                      len(failed_ids), total, outfile, failed_ids)

    return failed_ids
    
    
    # Generation never writes the POS lexicon.
    #
    # This used to call d.save_pos_cache() here.  That write was append-only --
    # "if a word is already in the cache on disk, do NOT overwrite it" -- so a
    # wrong tag, once written, was permanent: no amount of re-running could
    # correct it, and the run that produced it fed the next run's candidate
    # pools.  Measured on the shipped German cache, 14 of 17 proper nouns that
    # leaked into a sample run were tagged ADV/VERB/NUM/ADJ there while the
    # current tagger calls every one of them PROPN.
    #
    # The lexicon is now a read-only input, rebuilt deliberately by
    # `python tools/build_lexicon.py --lang de`.

