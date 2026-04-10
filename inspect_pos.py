import sys
import os
import importlib
from set_params import set_params
from input import read_input
from sentence_set import _get_nlp_model, strip_punct

def inspect_pos(params_file="params_de.txt", input_file=None):
    params = set_params(params_file)
    lang = params.get('lang', 'de')
    
    if input_file is None:
        # Try to find a likely input file
        for f in ['sentences_de.txt', 'input.txt', 'sentences.txt']:
            if os.path.exists(f):
                input_file = f
                break
    
    if not input_file or not os.path.exists(input_file):
        print(f"Error: Input file not found. Please provide one.")
        return

    print(f"\n>>> LOADING NLP MODEL ({lang})...")
    nlp = _get_nlp_model(lang, params)
    
    # Initialize dictionary to access our new heuristic
    dict_module = importlib.import_module(params.get("dictionary_loc", "wordfreq_distractor"))
    dict_class = getattr(dict_module, params.get("dictionary_class", "wordfreq_German_zipf_dict"))
    d = dict_class(params)

    print(f">>> READING INPUT: {input_file}")
    sents = read_input(input_file)
    
    print(f"\n{'TOKEN':<20} | {'STANZA POS':<12} | {'HEURISTIC IS_NOUN':<18} | {'CONSENSUS'}")
    print("-" * 75)

    for ss in sents.values():
        for sentence in ss.sentences:
            for word in sentence.text.split():
                clean = strip_punct(word)
                if not clean: continue
                
                # 1. Stanza POS
                stanza_pos = "N/A"
                if nlp and nlp != "FARASA_DELEGATE":
                    try:
                        doc = nlp(clean)
                        stanza_pos = doc.sentences[0].words[0].upos if doc.sentences else "?"
                    except Exception:
                        stanza_pos = "ERR"
                
                # 2. Heuristic
                is_noun_h = d.has_titlecase_variant(clean.lower())
                
                # 3. Consensus
                consensus = "NOUN" if (stanza_pos == 'NOUN' or is_noun_h) else "NON-NOUN"
                if stanza_pos == 'PROPN': consensus = "PROPN (NOUN)"

                print(f"{word:<20} | {stanza_pos:<12} | {str(is_noun_h):<18} | {consensus}")
            print("-" * 75)

if __name__ == "__main__":
    p_file = sys.argv[1] if len(sys.argv) > 1 else "params_de.txt"
    i_file = sys.argv[2] if len(sys.argv) > 2 else None
    inspect_pos(p_file, i_file)
