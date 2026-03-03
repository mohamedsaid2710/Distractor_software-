#!/usr/bin/env python3

import argparse
import os
import sys

# Keep CLI output focused on pipeline errors instead of TF backend noise.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from main import run_stuff


def main():
    parser = argparse.ArgumentParser(description="Run distractor generation")
    parser.add_argument('-i', '--input', dest='input', required=True, help='Input file')
    parser.add_argument('-o', '--output', dest='output', required=True, help='Output file')
    parser.add_argument('-p', '--parameters', type=str, default='params_en.txt', help='Parameters file (default: params_en.txt)')
    parser.add_argument('-f', '--format', choices=['ibex', 'delim'], default='delim', help='Output format')
    args = parser.parse_args()

    try:
        run_stuff(args.input, args.output, parameters=args.parameters, outformat=args.format)
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
