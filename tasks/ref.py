from collections import namedtuple
import numpy as np

Batch = namedtuple("Batch", ["target", "distractor", "left", "right", "label"])

class RefTask(object):
    def __init__(self):
        self.targets = np.load("data/target.npy")
        self.distractors = np.load("data/distractor.npy")
        assert self.targets.shape == self.distractors.shape
        self.n_examples, self.n_features = self.targets.shape
        self.random = np.random.RandomState(0)

    def get_batch(self, batch_size):
        indices = [np.random.randint(self.n_examples) for _ in
                range(batch_size)]
        target = []
        distractor = []
        left = []
        right = []
        label = []
        for i in indices:
            t = self.targets[i]
            d = self.distractors[i]
            target.append(t)
            distractor.append(d)
            if self.random.rand() < 0.5:
                left.append(t)
                right.append(d)
                label.append(0)
            else:
                left.append(d)
                right.append(t)
                label.append(1)

        return Batch(target, distractor, left, right, label)
