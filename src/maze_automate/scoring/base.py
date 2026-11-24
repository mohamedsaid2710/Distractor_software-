
class lang_model:
    """Parent class for the language models"""

    def __init__(self):
        """Do whatever set-up it takes"""
        pass


    def tokenize(self,word):
        """ returns a list of tokens according to the models desired tokenization scheme"""
        pass

    def empty_sentence(self):
        """Initialize a new sentence -- starter hidden state etc"""
        pass

    def update(self, hidden, word):
        """Append the next word to the context; returns (new context, None)."""
        pass

    def get_surprisal_from_hidden(self, hidden, word):
        """Surprisal (bits) of `word` after context `hidden`, summed over all
        of the word's subtokens."""
        pass

    def get_surprisal_batch_from_hidden(self, hidden, words, batch_size=None):
        """Given a hidden state and a list of words, 
        return a list of surprisals. 
        Base implementation is sequential for compatibility."""
        return [self.get_surprisal_from_hidden(hidden, w) for w in words]
