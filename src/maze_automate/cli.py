#!/usr/bin/env python3

import argparse
import os
import sys
import time
import traceback

# Keep CLI output focused on pipeline errors instead of TF backend noise.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Silence noisy PyTorch/HuggingFace warnings (e.g. UNEXPECTED bias tables)
import transformers
transformers.logging.set_verbosity_error()

# Silence noisy dependency loggers
import logging
logging.getLogger().setLevel(logging.WARNING)

from maze_automate.pipeline import run_stuff

# Exit codes.  2 is argparse's own code for a malformed command line.
EXIT_UNEXPECTED = 1
EXIT_PARAMS = 3
EXIT_INPUT = 4
EXIT_ITEMS_FAILED = 5
EXIT_SETUP = 6          # model loading, missing package, missing model/data file
EXIT_OUTPUT = 7         # output folder missing or not writable
EXIT_INTERRUPTED = 130  # Ctrl-C, shell convention


def fail(code, what, message):
    """Print one labelled error line (plus its underlying cause) and exit."""
    print(f"ERROR [{what}]: {message}", file=sys.stderr)
    cause = getattr(message, "__cause__", None)
    if cause is not None and str(cause):
        print(f"  caused by {type(cause).__name__}: {cause}", file=sys.stderr)
    sys.exit(code)


def print_splash(lang, model, nlp):
    banner = r"""
 ____  _     _                  _                  
|  _ \(_)___| |_ _ __ __ _  ___| |_ ___  _ __ ___  
| | | | / __| __| '__/ _` |/ __| __/ _ \| '__/ __| 
| |_| | \__ \ |_| | | (_| | (__| || (_) | |  \__ \ 
|____/|_|___/\__|_|  \__,_|\___|\__\___/|_|  |___/ 
             - S o f t w a r e -"""

    splash = f"""\033[96m{banner}
\033[0m                                                                 
=================================================================
 \033[93m» Target Language \033[0m: {lang}
 \033[93m» Model chosen \033[0m: {model}
 \033[93m» Distractor NLP  \033[0m: {nlp}
=================================================================
\033[92m[ System Online. Initializing pipeline... ]\033[0m
"""
    print(splash, flush=True)

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run distractor generation")

    # Flags only, all required. Positional arguments used to be accepted too,
    # and mixing the two styles could drop the params file into the wrong slot
    # and silently fall back to config/params_en.txt (English model on German input).
    parser.add_argument('-i', '--input', dest='input_path', required=True,
                        help='Input file')
    parser.add_argument('-o', '--output', dest='output_path', required=True,
                        help='Output file')
    parser.add_argument('-p', '--parameters', dest='params_path', required=True,
                        help='Parameters file (no default: it selects the language)')
    parser.add_argument('-f', '--format', choices=['ibex', 'delim'], default='delim',
                        help='Output format (default: delim)')
    parser.add_argument('--allow-tagger-fallback', action='store_true', default=None,
                        help='If the POS lexicon is missing, tag candidates with the '
                             'tagger without asking (needed for non-interactive runs)')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    params_path = args.params_path

    # Check the three paths up front so a missing file is reported by its
    # role (input / params / output folder), not as a bare path from deep
    # inside the pipeline.
    if not os.path.isfile(params_path):
        fail(EXIT_PARAMS, "params file", f"not found: {params_path}")
    if not os.path.isfile(input_path):
        fail(EXIT_INPUT, "input file", f"not found: {input_path}")
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if not os.path.isdir(out_dir):
        fail(EXIT_OUTPUT, "output file", f"folder does not exist: {out_dir}")

    from maze_automate.params import set_params, ParamsError
    from maze_automate.input import InputFormatError
    from maze_automate.pipeline import LexiconError
    from maze_automate.semantic_filter import SemanticFilterError
    from maze_automate.selection import TaggerError

    try:
        params = set_params(params_path)
    except ParamsError as e:
        fail(EXIT_PARAMS, "params file", e)

    from maze_automate.params import LANGUAGES
    lang_str = LANGUAGES[params["language"]]  # required and validated

    # hf_model_name is the key that actually loads the model
    # (scoring/hf_scorer.py); without it the scorer class's DEFAULT_MODEL is used.
    model_str = params.get("hf_model_name", "(default of model_class)")
    if lang_str == "German": nlp_str = "Stanza"
    elif lang_str == "Arabic": nlp_str = "Stanza"
    else: nlp_str = "SpaCy"

    print_splash(lang_str, model_str, nlp_str)

    try:
        failed_ids = run_stuff(input_path, output_path,
                               parameters=params_path, outformat=args.format,
                               allow_tagger_fallback=args.allow_tagger_fallback)
    except ParamsError as e:
        fail(EXIT_PARAMS, "params file", e)
    except LexiconError as e:
        fail(EXIT_SETUP, "POS lexicon", e)
    except SemanticFilterError as e:
        fail(EXIT_SETUP, "semantic filter", e)
    except TaggerError as e:
        fail(EXIT_SETUP, "sentence tagger", e)
    except InputFormatError as e:
        fail(EXIT_INPUT, "input file", e)
    except ImportError as e:
        fail(EXIT_SETUP, "missing package", e)
    except PermissionError as e:
        fail(EXIT_OUTPUT, "permission denied", f"{e.filename}: {e.strerror}")
    except FileNotFoundError as e:
        fail(EXIT_SETUP, "missing file", e)
    except RuntimeError as e:
        # Raised with a descriptive message by the model loaders.
        fail(EXIT_SETUP, "model loading", e)
    except KeyboardInterrupt:
        fail(EXIT_INTERRUPTED, "interrupted",
             f"stopped by user; {output_path} is INCOMPLETE")
    except Exception as e:
        traceback.print_exc()
        fail(EXIT_UNEXPECTED, f"unexpected {type(e).__name__}", e)

    elapsed = time.time() - start_time
    print(f"\n>>> [Done] Total run time: {elapsed / 60:.2f} minutes.")
    if failed_ids:
        # A partially-generated stimulus set must not look like a clean run
        # to a shell script or a CI job.  Each failure's traceback is printed
        # above, under its item ID.
        fail(EXIT_ITEMS_FAILED, "items failed",
             f"{len(failed_ids)} item(s) are missing from {output_path}: {failed_ids}")


if __name__ == '__main__':
    main()
