from data.color import munroecorpus
from ref import RefTask

import numpy as np

class ColorRefTask(RefTask):
    def __init__(self):
        super(ColorRefTask, self).__init__()

        corpus = munroecorpus.get_training_handles()
        self.vocab = {"_": 0}
        self.reverse_vocab = {0: "_"}
        self.sentences = []
        self.names = []
        self.colors = []
        self.max_sentence_len = 0
        for name in corpus[0]:
            words = name.split(" ")
            toks = []
            for word in words:
                if word not in self.vocab:
                    index = len(self.vocab)
                    self.vocab[word] = index
                    self.reverse_vocab[index] = word
                toks.append(self.vocab[word])
            self.max_sentence_len = max(self.max_sentence_len, len(toks))
            files = [c[name] for c in corpus]
            colors = np.array([munroecorpus.open_datafile(f) for f in files]).T

            for color in colors:
                self.colors.append(color)
                self.names.append(name)
                self.sentences.append(toks)

        self.colors = np.asarray(self.colors)
        self.colors[:, 0] /= 256
        self.colors[:, 1:] /= 100

        self.n_examples = len(self.colors)
        self.n_features = 6

    def get_pair(self):
        i1, i2 = self.random.randint(self.n_examples, size=2)
        return self.colors[i1], self.colors[i2]
