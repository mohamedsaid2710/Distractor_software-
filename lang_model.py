
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
        """Given the model representation (=hidden state) and the next word (not tokenized)
        returns new hidden state (at end of adding word)
        and probability distribution of next words at end of addition"""
        pass

    def get_surprisal(self,surprisals,word):
        """Given a probability distribution, and a word
        Return its surprisal (bits), or use -1 as error/surprisal not trusted  code"""
        pass

    def get_surprisal_batch_from_hidden(self, hidden, words, batch_size=None):
        """Given a hidden state and a list of words, 
        return a list of surprisals. 
        Base implementation is sequential for compatibility."""
        return [self.get_surprisal_from_hidden(hidden, w) for w in words]
