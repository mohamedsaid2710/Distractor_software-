
class Repeatcounter:
    """Keeps track of how many times each distractor has been used so far in the entire set of all items.
    Provides a list of those that are now banned b/c they have been used too much, and counts the rest.
    Disallows more than max number. If max is 0, no max is enforced."""

    def __init__(self, max_count):
        """initializes counter"""
        self.max = max_count
        if max_count == 0:
            self.limit = False
        else:
            self.limit = True
        self.distractors = dict()
        self.banned = []

    def increment(self, word):
        """adds a new repeat of word to the list, if this puts it up to max, adds it to banned"""
        word_key = word.lower()
        if word_key in self.distractors.keys():
            self.distractors[word_key] += 1
        else:
            self.distractors[word_key] = 1
        
        # Don't ban short words (3 or 4 letters) from repeating
        if self.limit and len(word_key) > 4:
            if self.distractors[word_key] >= self.max:
                self.banned.append(word_key)

