import logging


def strip_punct(word):
    '''take a word, return word with start and end punctuation removed'''
    for i in range(len(word)):
        if word[i].isalnum():
            break
    for j in range(len(word) - 1, -1, -1):
        if word[j].isalnum():
            break
    word = word[i:j + 1]
    return word

def copy_punct(word, distractor):
    """Copy leading/trailing punctuation from *word* onto *distractor*.

    Casing is NOT modified — that is handled separately by the
    language-specific normalization functions.
    """
    for i in range(len(word)):
        if word[i].isalnum():
            break
    prefix = word[0:i]
    for j in range(len(word) - 1, -1, -1):
        if word[j].isalnum():
            break
    suffix = word[j + 1:]
    return prefix + distractor + suffix
