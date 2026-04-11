import logging
import importlib
from set_params import set_params
from limit_repeats import Repeatcounter
from input import read_input
from output import save_ibex, save_delim, append_results
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
        pre_tag_all_distractors(sents, d, threshold_func, params, force_refresh=True)

    
    # PRE-CLEAR OUTFILE to allow incremental appending
    with open(outfile, 'w', encoding='utf-8') as f:
        pass

    total = len(sents)
    executed_ids = []
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
            executed_ids.append(ss_id)
            print(f"    ✅ Success: Item {ss_id} saved to {outfile}")
            
        except Exception as e:
            print(f"    ❌ ERROR on Item {ss_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
        
        ss.clean_up()
    
    print(f"\n>>> GENERATION COMPLETE. Processed {len(executed_ids)}/{total} items.")
    print(f">>> Successfully saved IDs: {executed_ids}")
    
    
    # FINAL CACHE SAVE: Persist all tagged words to disk
    if hasattr(d, 'save_pos_cache'):
        d.save_pos_cache()


def pre_tag_all_distractors(sents, d, threshold_func, params, force_refresh=True):
    """Proactively harvests and batch-tags potential distractor candidates
    for the entire experiment in a single Stanza batch run.

    force_refresh=True ensures that even if a word is in the cache, it is
    re-verified to purge messy legacy tags.
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
                # STRIP PUNCTUATION: Ensure we only tag the base word 
                # (prevents cache pollution like 'angereist.')
                stripped = re.sub(r"[^\w\s\u0600-\u06FFäöüÄÖÜß]", "", word)
                if not stripped: continue
                # Lowercase for canonical cache entry
                word_clean = stripped.lower()
                
                # Get thresholds for this word
                try:
                    min_l, max_l, min_f, max_f = threshold_func([word], params)
                    target_freq = (min_f + max_f) / 2
                    
                    # Fetch candidates of the exact required length
                    pool = d.get_best_frequency_pool(len(word_clean), target_freq, n=100)
                    for cand in pool:
                        all_candidates.add(cand.lower())
                except Exception:
                    continue
    
    if all_candidates:
        print(f"    [PRE-TAG] Found {len(all_candidates)} unique candidates to verify.")
        # batch_tag_words handles deduplication and cache-checking internally
        d.batch_tag_words(list(all_candidates), params, force_refresh=force_refresh)
        print(">>> [PRE-TAG] Parallel tagging complete.\n", flush=True)


