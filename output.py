import csv
import os 
import json


def save_delim(outfile, all_sentences):
    '''Saves results to a file in semicolon delimited format
    basically same as the original input with another column for distractor sentence
    Arguments:
    outfile = location of a file to write to
    all_sentences: dictionary of sentence_set objects
    Returns: none
    will write a semicolon delimited file with
    column 1 = "tag"/condition copied over from item_to_info (from input file)
    column 2 = item number
    column 3 = good sentence
    column 4 = string of distractor words in order.
    column 5 = string of labels in order. '''
    with open(outfile, 'w+', encoding='utf-8', newline="") as f:
        writer=csv.writer(f,delimiter=";")
        for sentence_set in all_sentences.values():
            for sentence in sentence_set.sentences:
                writer.writerow([sentence.tag,sentence.id,sentence.word_sentence,sentence.distractor_sentence,sentence.label_sentence])


def save_ibex(outfile, all_sentences):
    '''Saves results to a file in ibex format
    File contents can be copied into the items list of a maze_ibex file
    Arguments:
    outfile = location of a file to write to
    all_sentences: dictionary of sentence_set objects
    Returns: none'''
    with open(outfile, 'w+', encoding='utf-8', newline='') as f:
        for sentence_set in all_sentences.values():
            for sentence in sentence_set.sentences:
                # write a JS-like tuple for Ibex Maze. Escape double quotes in text.
                s = sentence.word_sentence.replace('"', '\\"')
                d = sentence.distractor_sentence.replace('"', '\\"')
                # format: [[tag, id], "Maze", {s:"<sentence>", a:"<distractor>"}],
                f.write(f'[["{sentence.tag}", {repr(sentence.id)}], "Maze", {{s:"{s}", a:"{d}"}}], \n')


def _match_case(target: str, candidate: str) -> str:
    """Match capitalization pattern of target onto candidate (basic heuristics)."""
    if not candidate:
        return candidate
    if target.isupper():
        return candidate.upper()
    if target[0].isupper():
        # preserve remainder of candidate as-is (capitalize first char)
        return candidate[0].upper() + candidate[1:]
    return candidate.lower()

def write_output(outfile, results):
    """Write results ensuring distractor casing matches target tokens.
    results: iterable of dicts/tuples per existing repo format (adjusted to use _match_case)."""
    with open(outfile, 'w', encoding='utf-8') as fh:
        for row in results:
            # example: if row has target token and distractor string fields
            # keep original repo formatting, but apply case matching where appropriate
            # Attempt to be backwards compatible: check common keys
            if isinstance(row, dict):
                target = row.get('target', '')
                distractor = row.get('distractor', '')
                if target and distractor:
                    distractor = _match_case(target, distractor)
                    row['distractor'] = distractor
                fh.write(json.dumps(row, ensure_ascii=False) + '\n')
            else:
                # fallback: assume tuple/list where one field is target and another is distractor
                try:
                    # common tuple form: (id, target_word, distractor, ...)
                    lst = list(row)
                    # try heuristic: find first string token to treat as target/distractor
                    if len(lst) >= 3 and isinstance(lst[1], str) and isinstance(lst[2], str):
                        lst[2] = _match_case(lst[1], lst[2])
                    fh.write('\t'.join(map(str, lst)) + '\n')
                except Exception:
                    fh.write(str(row) + '\n')
