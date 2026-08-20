import csv
import json


def ibex_line(tag, item_id, sentence, distractors):
    """Format one PCIbex Maze item line.

    Every text field goes through json.dumps, which is a superset of what a JS
    string literal needs.  The previous `.replace('"', chr(92)+chr(34))` escaped
    quotes but not backslashes, so a target containing one closed the literal
    early and PCIbex parsed the remainder of the line as code -- corrupting the
    whole item list, not just that item.  An embedded newline (which a quoted
    CSV field can carry) produced an unterminated literal with the same effect.

    Output is byte-identical to the old format for text without backslashes,
    newlines or control characters, i.e. for all ordinary stimuli.
    """
    s = json.dumps(sentence, ensure_ascii=False)
    d = json.dumps(distractors, ensure_ascii=False)
    t = json.dumps(tag, ensure_ascii=False)
    return f'[[{t}, {repr(item_id)}], "Maze", {{s:{s}, a:{d}}}], \n'


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
                f.write(ibex_line(sentence.tag, sentence.id,
                                  sentence.word_sentence, sentence.distractor_sentence))



def append_results(outfile, sentence_set, outformat):
    '''Appends a single sentence_set's results to the specified outfile.
    Ensures that output is flushed so user can see it immediately.'''
    if outformat == "delim":
        with open(outfile, 'a', encoding='utf-8', newline="") as f:
            writer = csv.writer(f, delimiter=";")
            for sentence in sentence_set.sentences:
                writer.writerow([sentence.tag, sentence.id, sentence.word_sentence, 
                                 sentence.distractor_sentence, sentence.label_sentence])
            f.flush()
    elif outformat == "ibex":
        with open(outfile, 'a', encoding='utf-8', newline='') as f:
            for sentence in sentence_set.sentences:
                f.write(ibex_line(sentence.tag, sentence.id,
                                  sentence.word_sentence, sentence.distractor_sentence))
            f.flush()
