from batch import Batch
from data.color import munroecorpus

import numpy as np

class ColorRefTask(object):
    def __init__(self):
        self.random = np.random.RandomState(0)
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
        self.n_features = 3

    def get_batch(self, batch_size):
        indices = self.random.randint(self.n_examples, size=(batch_size, 2))
        target = []
        distractor = []
        left = []
        right = []
        label = []
        sentence = np.zeros((batch_size, self.max_sentence_len))
        i = 0
        for ti, di in indices:
            t = self.colors[ti]
            d = self.colors[di]
            s = self.sentences[ti]
            target.append(t)
            distractor.append(d)
            sentence[i, :len(s)] = s
            if self.random.rand() < 0.5:
                left.append(t)
                right.append(d)
                label.append(0)
            else:
                left.append(d)
                right.append(t)
                label.append(1)
            i += 1

        return Batch(target, distractor, left, right, sentence, label)

    def decode(self, tok_ids):
        return " ".join([self.reverse_vocab[i] for i in tok_ids])
