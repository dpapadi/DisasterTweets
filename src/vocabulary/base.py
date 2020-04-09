class Vocabulary():
    """ Vocabulary that creates a mapping from a word token to an index.
        Used for associating words with embedding indices
    """
    def __init__(self):
        self.PAD = 0
        self.index2word = {self.PAD: "PAD"}
        self.word2index = {"PAD": self.PAD}
        self.size = 1
        # store all OOV tokens and count how many times they appear
        self.oov = {}

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.size
            self.index2word[self.size] = word
            self.size += 1
        return self.size

    def add_list(self, voc_list):
        for word in voc_list:
            self.add_word(word)
        return self

    def get_word(self, index):
        try:
            return self.index2word[index]
        except Exception:
            return "Error: INDEX TOO BIG"

    def get_index(self, word):
        try:
            return self.word2index[word]
        except Exception:
            if word not in self.oov:
                self.oov[word] = 1
            else:
                self.oov[word] += 1
            return -1

    def reset_oov(self):
        """ reset Out Of Vocabulary dict
        """
        self.oov = {}
        return "1"
