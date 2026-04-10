import sys
import os
import importlib
import argparse
from set_params import set_params
from input import read_input
from sentence_set import _get_nlp_model, strip_punct

def run_inspection(nlp, d, sentence_text, source="Manual"):
    print(f"\n>>> SOURCE: {source}")
    print(f"{'TOKEN':<20} | {'STANZA POS':<12} | {'HEURISTIC IS_NOUN':<18} | {'CONSENSUS'}")
    print("-" * 75)
    
    tokens = sentence_text.split()
    for word in tokens:
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

def main():
    parser = argparse.ArgumentParser(description="German Maze POS Inspection Utility")
    parser.add_argument("--params", default="params_de.txt", help="Path to params file (default: params_de.txt)")
    parser.add_argument("--input", help="Path to input sentence file")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()

    if not os.path.exists(args.params):
        print(f"Warning: Params file '{args.params}' not found. Using defaults.")
        params = {'lang': 'de', 'dictionary_loc': 'wordfreq_distractor', 'dictionary_class': 'wordfreq_German_zipf_dict'}
    else:
        params = set_params(args.params)
    
    lang = params.get('lang', 'de')
    
    print(f"\n>>> INITIALIZING POS DEBUGGER ({lang})...")
    nlp = _get_nlp_model(lang, params)
    
    # Initialize dictionary to access heuristic
    try:
        dict_module = importlib.import_module(params.get("dictionary_loc", "wordfreq_distractor"))
        dict_class = getattr(dict_module, params.get("dictionary_class", "wordfreq_German_zipf_dict"))
        d = dict_class(params)
    except Exception as e:
        print(f"Error initializing dictionary: {e}")
        return

    # Mode 1: File Input
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found.")
        else:
            print(f">>> READING INPUT: {args.input}")
            sents = read_input(args.input)
            for ss in sents.values():
                for sentence in ss.sentences:
                    run_inspection(nlp, d, sentence.text, source=args.input)
    
    # Mode 2: Interactive (or if no input file and interactive flag or just no file)
    if args.interactive or (not args.input):
        print("\n=== INTERACTIVE POS INSPECTION MODE ===")
        print("Type a sentence and press Enter to see POS analysis. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                user_input = input("\nSentence > ").strip()
                if user_input.lower() in ['exit', 'quit']:
                    break
                if not user_input:
                    continue
                run_inspection(nlp, d, user_input, source="User Interaction")
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
