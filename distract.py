#!/usr/bin/env python3

import argparse
import os
import sys
import time

# Keep CLI output focused on pipeline errors instead of TF backend noise.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from main import run_stuff


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run distractor generation")
    
    # Positional arguments for better tab-completion
    parser.add_argument('input_file', nargs='?', help='Input file (can also use -i)')
    parser.add_argument('output_file', nargs='?', help='Output file (can also use -o)')
    parser.add_argument('params_file', nargs='?', help='Parameters file (can also use -p)')
    
    # Optional flags for backwards compatibility
    parser.add_argument('-i', '--input', dest='input_flag', help='Input file')
    parser.add_argument('-o', '--output', dest='output_flag', help='Output file')
    parser.add_argument('-p', '--parameters', dest='params_flag', help='Parameters file')
    
    parser.add_argument('-f', '--format', choices=['ibex', 'delim'], default='delim', help='Output format')
    args = parser.parse_args()

    # Resolve arguments: flags take precedence, then positional
    input_path = args.input_flag or args.input_file
    output_path = args.output_flag or args.output_file
    params_path = args.params_flag or args.params_file or 'params_en.txt'

    if not input_path or not output_path:
        parser.error("Both input and output files are required.")

    try:
        run_stuff(input_path, output_path, parameters=params_path, outformat=args.format)
        
        elapsed = time.time() - start_time
        print(f"\n>>> [Done] Total run time: {elapsed / 60:.2f} minutes.")
    except ValueError as e:
        import traceback
        traceback.print_exc()
        print('ERROR: input parsing failed:', e, file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError as e:
        print('ERROR: missing file:', e, file=sys.stderr)
        sys.exit(3)
    except RuntimeError as e:
        print('ERROR:', e, file=sys.stderr)
        sys.exit(4)


if __name__ == '__main__':
    main()
