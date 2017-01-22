from ref import RefTask

from collections import defaultdict
import csv
import numpy as np
import pickle

N_FEATURES = 256

class BirdsRefTask(RefTask):
    def __init__(self):
        super(BirdsRefTask, self).__init__(N_FEATURES)
        with open("data/birds/CUB_feature_dict.p") as feature_f:
            features = pickle.load(feature_f)
        self.random = np.random.RandomState(0)
        n_raw_features, = features.values()[0].shape
        projector = self.random.randn(N_FEATURES, n_raw_features)
        proj_features = {}
        m1 = np.zeros(N_FEATURES)
        m2 = np.zeros(N_FEATURES)
        for k, v in features.items():
            proj = np.dot(projector, v)
            m1 += proj
            m2 += proj ** 2
            proj_features[k] = proj
        m1 /= len(features)
        m2 /= len(features)
        std = np.sqrt(m2 - m1 ** 2)
        for v in proj_features.values():
            v -= m1
            v /= std
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
                anns[key] = desc.split()
        counts = defaultdict(lambda: 0)
        for desc in anns.values():
            for word in desc:
                counts[word] += 1
        freq_words = sorted(counts.items(), key=lambda p: -p[1])
        self.vocab = {"_": 0, "UNK": 1}
        self.reverse_vocab = {0: "_", 1: "UNK"}
        for word, count in freq_words[:100]:
            i = len(self.vocab)
            self.vocab[word] = i
            self.reverse_vocab[i] = word
        self.n_vocab = len(self.vocab)
        self.descs = {}
        self.max_desc_len = 0
        for k, v in anns.items():
            out = []
            for word in v:
                if word in self.vocab:
                    out.append(self.vocab[word])
                else:
                    out.append(self.vocab["UNK"])
            #out = [np.random.randint(2)+2] * 2
            self.descs[k] = out
            self.max_desc_len = max(self.max_desc_len, len(out))

        ### self.vocab = {"UNK": 0, "foo": 1, "baz": 2}
        ### self.reverse_vocab = {0: "UNK", 1: "foo", 2: "baz"}
        ### self.descs = {}
        ### for k in self.keys:
        ###     #if k in self.folds["train"]:
        ###     #    i = 1
        ###     #else:
        ###     #    i = 2
        ###     i = 1 + self.random.randint(2)
        ###     self.descs[k] = [i] * 2
        ### self.max_desc_len = 2
        ### self.n_vocab = 3

        self.n_examples = len(self.keys)

    def get_pair(self, fold):
        fold_keys = self.folds[fold]
        i1, i2 = self.random.randint(len(fold_keys), size=2)
        k1, k2 = fold_keys[i1], fold_keys[i2]
        f1, f2 = self.features[k1], self.features[k2]
        desc = self.descs[k1]

        #f1 = np.zeros(f1.shape)
        #f2 = np.zeros(f2.shape)
        #f1[0] = desc[0] - 2
        #f2[0] = 1 - (desc[0] - 2)

        #if len(desc) - 1 == 0:
        #    i_bigram = 0
        #else:
        #    i_bigram = self.random.randint(len(desc)-1)
        #bigram = desc[i_bigram:i_bigram+2]
        ### bigram = self.descs[k1]
        return (f1, f2, desc, k1, k2)

    def visualize(self, state, agent):
        url_template = "http://tomato.banatao.berkeley.edu/jda/codes/birds/CUB_200_2011/images/%s"
        url1 = url_template % state.left_data
        url2 = url_template % state.right_data
        if agent == 0 and state.target == 1:
            url1, url2 = url2, url1
        html_template = "<img src='%s'>"
        return (html_template % url1) + (html_template % url2)

    def pp(self, indices):
        return " ".join([self.reverse_vocab[i] for i in indices])
