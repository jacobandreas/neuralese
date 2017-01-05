from struct import Batch

import numpy as np

class DummyRefTask(object):
    def __init__(self):
        a = [1, 0]
        b = [0, 1]
        c = [1, 1]
        self.examples = [
            (a, b, [0]),
            (a, c, [0]),
            (b, a, [1]),
            (b, c, [1]),
            (c, a, [1]),
            (c, b, [0])
        ]
        self.n_examples = len(self.examples)
        self.n_features = 2
        self.max_sentence_len = 1
        self.vocab = {"left": 0, "right": 1}
        self.random = np.random.RandomState(0)

    def get_batch(self, batch_size):
        indices = self.random.randint(self.n_examples, size=batch_size)
        target = []
        distractor = []
        left = []
        right = []
        label = []
        sentence = []
        for i in indices:
            t, d, s = self.examples[i]
            target.append(t)
            distractor.append(d)
            sentence.append(s)
            if self.random.rand() < 0.5:
                left.append(t)
                right.append(d)
                label.append(0)
            else:
                left.append(d)
                right.append(t)
                label.append(1)

        return Batch(target, distractor, left, right, sentence, label)

    def decode(self, tok_ids):
        return " ".join([["left", "right"][i] for i in tok_ids])
