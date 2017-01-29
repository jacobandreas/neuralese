from ref import RefTask

from collections import defaultdict
import csv
import numpy as np
import pickle

N_FEATURES = 256
STOP = [
    "a", "an", "the", "this", "it", "its", "and", "is", "are", "has", "have",
    "with", "on", "in"
]

class BirdsRefTask(RefTask):
    def __init__(self):
        super(BirdsRefTask, self).__init__(N_FEATURES)
        #with open("data/birds/CUB_feature_dict.p") as feature_f:
        with open("data/birds/labels.p") as feature_f:
            features = pickle.load(feature_f)
        self.random = np.random.RandomState(0)
        n_raw_features, = features.values()[0].shape
        projector = self.random.randn(N_FEATURES, n_raw_features)
        proj_features = {}
        m1 = np.zeros(N_FEATURES)
        m2 = np.zeros(N_FEATURES)
        for k, v in features.items():
            #proj = np.dot(projector, v)
            proj = v
            m1 += proj
            m2 += proj ** 2
            proj_features[k] = proj
        m1 /= len(features)
        m2 /= len(features)
        std = np.sqrt(m2 - m1 ** 2)
        #for v in proj_features.values():
        #    v -= m1
        #    v /= std
        self.features = proj_features
        self.keys = sorted(self.features.keys())
        
        self.folds = {}
        for fold in ["train", "val", "test"]:
            with open("data/birds/%s.txt" % fold) as fold_f:
                self.folds[fold] = [line.strip() for line in fold_f]

        anns = {}
        with open("data/birds/cub_0917_5cap.tsv") as ann_f:
            reader = csv.reader(ann_f, delimiter="\t")
            header = reader.next()
            i_url = header.index("Input.image_url")
            i_desc = header.index("Answer.Description")
            for line in reader:
                url = line[i_url]
                desc = line[i_desc]
                key = "/".join(url.split("/")[-2:])
                desc = desc.lower().replace(".", "").replace(",", "")
                anns[key] = tuple(desc.split())
        counts = defaultdict(lambda: 0)
        for desc in anns.values():
            for i in range(len(desc) - 1):
                bigram = desc[i:i+2]
                counts[bigram] += 1
        freq_terms = sorted(counts.items(), key=lambda p: -p[1])
        freq_terms = [f for f in freq_terms if not any(w in STOP for w in f[0])]
        freq_terms = [f[0] for f in freq_terms]
        freq_terms = freq_terms[:50]

        self.vocab = {"_": 0, "UNK": 1}
        self.reverse_vocab = {0: "_", 1: "UNK"}
        self.lexicon = [[0]]
        for term in freq_terms:
            for word in term:
                if word in self.vocab:
                    continue
                index = len(self.vocab)
                self.vocab[word] = index
                self.reverse_vocab[index] = word
            self.lexicon.append([self.vocab[w] for w in term])

        discarded = []
        self.reps = {}
        for k, desc in anns.items():
            rep = np.zeros(len(self.lexicon))
            out = []
            for i_l, l in enumerate(self.lexicon):
                if all(self.reverse_vocab[w] in desc for w in l):
                    rep[i_l] = 1
                    out += l
            if len(out) == 0:
                discarded.append(k)
                continue
            assert rep.any()
            rep /= np.sum(rep)
            self.reps[k] = rep
        for k in discarded:
            del anns[k]
            for fold in self.folds.values():
                if k in fold:
                    fold.remove(k)

        self.empty_desc = np.zeros(len(self.lexicon))
        self.empty_desc[0] = 1

    def reset_test(self):
        self.random = np.random.RandomState(0)

    def get_pair(self, fold):
        fold_keys = self.folds[fold]
        i1, i2 = self.random.randint(len(fold_keys), size=2)
        k1, k2 = fold_keys[i1], fold_keys[i2]
        f1, f2 = self.features[k1], self.features[k2]
        rep = self.reps[k1]
        return (f1, f2, rep, k1, k2)

    def visualize(self, state, agent):
        url_template = "http://tomato.banatao.berkeley.edu/jda/codes/birds/CUB_200_2011/images/%s"
        url1 = url_template % state.left_data
        url2 = url_template % state.right_data
        if agent == 0 and state.target == 1:
            url1, url2 = url2, url1
        html_template = "<img src='%s'>"
        return (html_template % url1) + (html_template % url2)

    def turk_visualize(self, state, agent, loc):
        left = state.left_data
        right = state.right_data
        if agent == 0 and state.target == 1:
            left, right = right, left
        return left, right

    def pp(self, indices):
        return " ".join([self.reverse_vocab[i] for i in indices])
