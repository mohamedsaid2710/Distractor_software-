import logging
import importlib
from set_params import set_params
from limit_repeats import Repeatcounter
from input import read_input
from output import save_ibex, save_delim
import re


#these are the default values, but can be overridden by parameters file 
def run_stuff(infile, outfile, parameters="params_en.txt", outformat="delim"):
    """Takes an input file, and an output file location
    Does the whole distractor thing (according to specified parameters)
    Writes in outformat"""
    if outformat not in ["delim", "ibex"]:
        logging.error("outfile format not understood")
        raise ValueError
    params = set_params(parameters)
    sents = read_input(infile)
    dict_class = getattr(importlib.import_module(params.get("dictionary_loc", "wordfreq_distractor")),
                         params.get("dictionary_class", "wordfreq_English_zipf_dict"))
    d = dict_class(params)
    model_class = getattr(importlib.import_module(params.get("model_loc", "models.english_code.model")),
                          params.get("model_class", "EnglishScorer"))
    m = model_class(params)
    threshold_func = getattr(importlib.import_module(params.get("threshold_loc", "wordfreq_distractor")),
                             params.get("threshold_name", "get_thresholds_en"))
    repeats=Repeatcounter(params.get("max_repeat", 0))
    
    # PROACTIVE PARALLEL TAGGING: Scan all sentences and tag likely candidates in one batch
    if params.get("proactive_tagging", True):
        pre_tag_all_distractors(sents, d, threshold_func, params)
    
    total = len(sents)
    for i, ss in enumerate(sents.values(), 1):
        # Get tag from the first sentence in the set (most sets only have one)
        tag = ss.sentences[0].tag if ss.sentences else "None"
        print(f"\n>>> [{i}/{total}] Processing Sentence: {tag} (ID: {ss.id})")
        
        ss.do_model(m)
        ss.do_surprisals(m)
        ss.make_labels(params)
        ss.do_distractors(m, d, threshold_func, params, repeats)
        ss.clean_up()
    
    if outformat == "delim":
        save_delim(outfile, sents)
    else:
        save_ibex(outfile, sents)
    
    # FINAL CACHE SAVE: Persist all tagged words to disk
    if hasattr(d, 'save_pos_cache'):
        d.save_pos_cache()


def pre_tag_all_distractors(sents, d, threshold_func, params):
    """Proactively harvests and batch-tags potential distractor candidates
    for the entire experiment in a single Stanza batch run.
    """
    if not hasattr(d, 'batch_tag_words') or getattr(d, 'nlp_sp', None) is None:
        return

    print("\n>>> [PRE-TAG] Harvesting candidate pool for parallel tagging...", flush=True)
    all_candidates = set()
    
    # Helper to strip punctuation for length matching
    def _strip(w):
        return re.sub(r"[^\w\s\u0600-\u06FFäöüÄÖÜß]", "", w)

    for ss in sents.values():
        for sentence in ss.sentences:
            # Determine which words need proactive distractors
            words_to_process = list(sentence.words)
            if bool(params.get("first_token_placeholder", True)) and words_to_process:
                words_to_process = words_to_process[1:]
                
            for word in words_to_process:
                stripped = _strip(word)
                if not stripped: continue
                
                # Get thresholds for this word
                try:
                    min_l, max_l, min_f, max_f = threshold_func([word], params)
                    target_freq = (min_f + max_f) / 2
                    
                    # Fetch candidates of the exact required length
                    pool = d.get_best_frequency_pool(len(stripped), target_freq, n=100)
                    for cand in pool:
                        all_candidates.add(cand.lower())
                except Exception:
                    continue
    
    if all_candidates:
        print(f"    [PRE-TAG] Found {len(all_candidates)} unique candidates to verify.")
        # batch_tag_words handles deduplication and cache-checking internally
        d.batch_tag_words(list(all_candidates), params)
        print(">>> [PRE-TAG] Parallel tagging complete.\n", flush=True)

