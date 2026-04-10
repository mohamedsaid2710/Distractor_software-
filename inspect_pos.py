import sys
import os
import importlib
import argparse
from set_params import set_params
from input import read_input
from sentence_set import _get_nlp_model, strip_punct

def run_inspection(nlp, d, sentence_text, source="Manual"):
    print(f"\n>>> SOURCE: {source}")
    print(f"SENTENCE: {sentence_text}")
    print(f"{'INDEX':<5} | {'TOKEN':<15} | {'STANZA POS':<12} | {'IS_NOUN'}")
    print("-" * 60)
    
    if nlp and nlp != "FARASA_DELEGATE":
        try:
            doc = nlp(sentence_text)
            global_idx = 0
            for sent_idx, sentence in enumerate(doc.sentences):
                for word_obj in sentence.words:
                    token = word_obj.text
                    pos = word_obj.upos
                    is_noun = pos in ('NOUN', 'PROPN')
                    print(f"{global_idx:<5} | {token:<15} | {pos:<12} | {is_noun}")
                    global_idx += 1
        except Exception as e:
            print(f"Stanza Error: {e}")
    else:
        # Fallback for non-Stanza or no NLP
        tokens = sentence_text.split()
        for i, word in enumerate(tokens):
            print(f"{i:<5} | {word:<15} | {'N/A':<12} | {'N/A'}")
    
    print("-" * 60)

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
                    run_inspection(nlp, d, sentence.word_sentence, source=args.input)
    
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
