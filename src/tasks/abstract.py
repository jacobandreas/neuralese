from data.abstract import corpus
from ref import RefTask

import numpy as np

class AbstractRefTask(RefTask):
    def __init__(self):
        super(AbstractRefTask, self).__init__()
        scenes, _, _, vocab, freq = corpus.load_abstract()
        self.scenes = scenes
        self.n_features = scenes[0].features.size * 2
        #self.n_features = 2
        self.vocab = vocab
        self.freq_vocab = freq
        self.n_vocab = len(vocab)
        self.reverse_vocab = {}
        for k, v in vocab.items():
            assert v not in self.reverse_vocab
            self.reverse_vocab[v] = k
        self.random = np.random.RandomState(0)
        self.n_examples = len(self.scenes)
        self.max_desc_len = max(len(s.description) for s in scenes)

    def get_pair(self):
        i1, i2 = self.random.randint(self.n_examples, size=2)
        s1, s2 = self.scenes[i1], self.scenes[i2]
        return (s1.features, s2.features, s1.description, s1.image_id, 
                s2.image_id)

    def visualize(self, state, agent):
        url_template = "http://fromage.banatao.berkeley.edu/pragma/data/abstract/RenderedScenes/Scene%s.png"
        url1 = url_template % state.left_data
        url2 = url_template % state.right_data
        if agent == 0 and state.target == 1:
            url1, url2 = url2, url1

        html_template = "<img src='%s'>"
        return (html_template % url1) + "\n" + (html_template % url2)

    def pp(self, indices):
        return " ".join([self.reverse_vocab[i] for i in indices])
