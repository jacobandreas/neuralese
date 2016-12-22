from batch import Batch

from collections import defaultdict
import numpy as np
import refer

class ImageRefTask(object):
    def __init__(self):
        self.targets = np.load("data/image/target.npy")
        self.target_ids = np.load("data/image/target_id.npy")
        self.distractors = np.load("data/image/distractor.npy")
        assert self.targets.shape == self.distractors.shape
        self.n_examples, self.n_features = self.targets.shape
        self.random = np.random.RandomState(0)
        self.refer = refer.REFER("../refer/data")

        counts = defaultdict(lambda: 0)
        self.max_sentence_len = 0
        for ref in self.refer.Refs.values():
            for sent in ref["sentences"]:
                for tok in sent["tokens"]:
                    counts[tok] += 1
                self.max_sentence_len = max(
                        self.max_sentence_len, len(sent["tokens"]))
        frequent_words = sorted(counts.items(), key=lambda x: -x[1])
        self.vocab = {"_": 0}
        for word, count in frequent_words[:1000]:
            self.vocab[word] = len(self.vocab)
        self.reverse_vocab = {i: w for w, i in self.vocab.items()}

    def decode(self, tok_ids):
        return " ".join(self.reverse_vocab[i] for i in tok_ids)

    def get_batch(self, batch_size):
        indices = [np.random.randint(self.n_examples) for _ in
                range(batch_size)]
        target = []
        distractor = []
        left = []
        right = []
        label = []
        sentence = []
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

            ref = self.refer.Refs[self.target_ids[i]]
            i_sent = self.random.choice(len(ref["sentences"]))
            sent = ref["sentences"][i_sent]
            toks = [self.vocab[t] for t in sent["tokens"] if t in self.vocab]
            toks += [0] * (self.max_sentence_len - len(toks))
            sentence.append(toks)

        return Batch(target, distractor, left, right, sentence, label)
