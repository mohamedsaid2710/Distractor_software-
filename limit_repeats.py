
class Repeatcounter:
    """Keeps track of how many times each distractor has been used so far in the entire set of all items.
    Provides a list of those that are now banned b/c they have been used too much, and counts the rest.
    Disallows more than max number. If max is 0, no max is enforced."""

    def __init__(self, max_count, exempt_max_len=4):
        """initializes counter.

        `exempt_max_len` is the length at or below which a word is exempt from
        `max_repeat`.  This used to be a hardcoded `len(word_key) > 4` with no
        way to see or change it, so `max_repeat: 1` was silently unenforced for
        every word of four characters or fewer -- which is most distractors for
        short function-word targets.  Set it to 0 to enforce the limit at all
        lengths.
        """
        self.max = max_count
        if max_count == 0:
            self.limit = False
        else:
            self.limit = True
        self.exempt_max_len = int(exempt_max_len)
        self.distractors = dict()
        self.banned = []

    def increment(self, word):
        """adds a new repeat of word to the list, if this puts it up to max, adds it to banned"""
        word_key = word.lower()
        if word_key in self.distractors.keys():
            self.distractors[word_key] += 1
        else:
            self.distractors[word_key] = 1

        if self.limit and len(word_key) > self.exempt_max_len:
            if self.distractors[word_key] >= self.max:
                self.banned.append(word_key)
