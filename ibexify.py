#!/usr/bin/env python3

"""Convert delim output files into Ibex item lines."""

import argparse
import csv
import logging

from output import ibex_line


def ibexify(infile, outfile):
    """Convert semicolon-delimited distractor output into Ibex format."""
    with open(infile, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=";", quotechar='"')
        with open(outfile, "w+", encoding="utf-8", newline="") as out:
            for lineno, row in enumerate(reader, start=1):
                if not row:
                    continue
                if len(row) < 4:
                    logging.warning(
                        "Skipping line %d in %s: expected >=4 columns, got %d",
                        lineno,
                        infile,
                        len(row),
                    )
                    continue
                tag, item_id, sentence, distractors = row[0], row[1], row[2], row[3]
                out.write(ibex_line(tag, item_id, sentence, distractors))


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Convert a semicolon-delimited distractor file to Ibex lines."
    )
    parser.add_argument("-i", "--input", required=True, help="Input delim file path")
    parser.add_argument("-o", "--output", required=True, help="Output Ibex file path")
    return parser


def main():
    args = _build_parser().parse_args()
    ibexify(args.input, args.output)


if __name__ == "__main__":
    main()
