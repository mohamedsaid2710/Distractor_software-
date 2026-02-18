import logging
import importlib
from set_params import set_params
from limit_repeats import Repeatcounter
from input import read_input
from output import save_ibex, save_delim
from fix_output_casing import process_file
import os.path


def _should_apply_postcase(params):
    """Return True when language-specific post-casing should run."""
    apply_postcase = params.get("apply_postcase", None)
    if apply_postcase is not None:
        return bool(apply_postcase)
    model_loc = (params.get("model_loc", "") or "").lower()
    dict_cls = (params.get("dictionary_class", "") or "").lower()
    hf_name = (params.get("hf_model_name", "") or "").lower()
    return ("german" in model_loc) or ("german" in dict_cls) or ("dbmdz" in hf_name)

#these are the deafult values, but can be overridden by parameters file 
def run_stuff(infile, outfile, parameters="config/params.txt", outformat="delim"):
    """Takes an input file, and an output file location
    Does the whole distractor thing (according to specified parameters)
    Writes in outformat"""
    if outformat not in ["delim", "ibex"]:
        logging.error("outfile format not understood")
        raise ValueError
    params = set_params(parameters)
    sents = read_input(infile)
    dict_class = getattr(importlib.import_module(params.get("dictionary_loc", "wordfreq_distractor")),
                         params.get("dictionary_class", "wordfreq_English_dict"))
    d = dict_class(params)
    model_class = getattr(importlib.import_module(params.get("model_loc", "models.english_code.model")),
                          params.get("model_class", "EnglishScorer"))
    m = model_class(params)
    threshold_func = getattr(importlib.import_module(params.get("threshold_loc", "wordfreq_distractor")),
                             params.get("threshold_name", "get_thresholds_en"))
    repeats=Repeatcounter(params.get("max_repeat", 0))
    for ss in sents.values():
        ss.do_model(m)
        ss.do_surprisals(m)
        ss.make_labels()
        ss.do_distractors(m, d, threshold_func, params, repeats)
        ss.clean_up()
    
    ####This is a post_processing of the output file! It is intergerated in the pipeline, so there is no need to call it separately! 
    if outformat == "delim":
        save_delim(outfile, sents)
        if _should_apply_postcase(params):
            try:
                # Enforce German noun capitalization on distractors (safe in-place)
                process_file(outfile, outfile)
            except Exception as e:
                logging.warning("Post-case processing failed: %s", e)
    else:
        save_ibex(outfile, sents)
