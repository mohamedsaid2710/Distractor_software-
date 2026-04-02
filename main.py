import logging
import importlib
import torch
from set_params import set_params
from limit_repeats import Repeatcounter
from input import read_input
from output import save_ibex, save_delim


#these are the default values, but can be overridden by parameters file 
def run_stuff(infile, outfile, parameters="params_en.txt", outformat="delim"):
    """Takes an input file, and an output file location
    Does the whole distractor thing (according to specified parameters)
    Writes in outformat"""
    # Fix for Stanza loading in PyTorch 2.6+
    if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
        try:
            import numpy
            torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
        except Exception:
            pass

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
