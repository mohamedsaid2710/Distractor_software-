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
    # positional args for backwards compatibility
    parser.add_argument('pos_input', nargs='?', help='Input file (positional, for backwards compat)')
    parser.add_argument('pos_output', nargs='?', help='Output file (positional, for backwards compat)')
    # named args
    parser.add_argument('-i', '--input', dest='input', help='Input file')
    parser.add_argument('-o', '--output', dest='output', help='Output file')
    parser.add_argument('-p', '--parameters', type=str, default=None, help='Parameters file (default: config/params.txt)')
    parser.add_argument('-f', '--format', choices=['ibex', 'delim'], default='delim', help='Output format')
    args = parser.parse_args()

    # allow either positional or named input/output
    infile = args.input if args.input is not None else args.pos_input
    outfile = args.output if args.output is not None else args.pos_output
    if infile is None or outfile is None:
        parser.error('the following arguments are required: input output (positional) or -i/--input and -o/--output')

    try:
        params = args.parameters if args.parameters is not None else 'config/params.txt'
        run_stuff(infile, outfile, parameters=params, outformat=args.format)
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
