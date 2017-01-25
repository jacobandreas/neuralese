from data.color import munroecorpus
from ref import RefTask

import cairo
import colorsys
import numpy as np
import os

class ColorRefTask(RefTask):
    def __init__(self):
        n_features = 3
        super(ColorRefTask, self).__init__(n_features)

        corpus = munroecorpus.get_training_handles()
        self.vocab = {"_": 0}
        self.reverse_vocab = {0: "_"}
        self.sentences = []
        self.names = []
        self.colors = []
        self.max_desc_len = 0
        for name in corpus[0]:
            words = name.split(" ")
            toks = []
            for word in words:
                if word not in self.vocab:
                    index = len(self.vocab)
                    self.vocab[word] = index
                    self.reverse_vocab[index] = word
                toks.append(self.vocab[word])
            self.max_desc_len = max(self.max_desc_len, len(toks))
            files = [c[name] for c in corpus]
            colors = np.array([munroecorpus.open_datafile(f) for f in files]).T

            for color in colors:
                self.colors.append(color)
                self.names.append(name)
                self.sentences.append(toks)

        self.colors = np.asarray(self.colors)
        self.colors[:, 0] /= 360
        self.colors[:, 1:] /= 100

        self.n_examples = len(self.colors)
        self.n_vocab = len(self.vocab)

        self.image_counter = 0

    def get_pair(self, fold):
        i1, i2 = self.random.randint(self.n_examples, size=2)
        return self.colors[i1, :], self.colors[i2, :], self.sentences[i1], None, None

    def visualize(self, state, agent):
        h1 = state.left[0] * 360
        s1 = state.left[1] * 100
        l1 = state.left[2] * 100

        h2 = state.right[0] * 360
        s2 = state.right[1] * 100
        l2 = state.right[2] * 100

        block1 = '<span style="display: inline-block; width: 20px; height: 20px; background: hsl(%s, %s%%, %s%%); border: 2px solid #000"></span>' % (h1, s1, l1)
        block2 = '<span style="display: inline-block; width: 20px; height: 20px; background: hsl(%s, %s%%, %s%%); border: 2px solid #000"></span>' % (h2, s2, l2)
        if agent == 0 and state.target == 1:
            block1, block2 = block2, block1
        return block1 + "\n" + block2

    def turk_visualize(self, state, agent, loc):
        data1 = np.zeros((200, 200, 4), dtype=np.uint8)
        surf1 = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_ARGB32, 200, 200)
        ctx1 = cairo.Context(surf1)
        rgb1 = colorsys.hls_to_rgb(state.left[0], state.left[2], state.left[1])
        ctx1.set_source_rgb(*rgb1)
        ctx1.fill()

        data2 = np.zeros((200, 200, 4), dtype=np.uint8)
        surf2 = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_ARGB32, 200, 200)
        ctx2 = cairo.Context(surf1)
        rgb2 = colorsys.hls_to_rgb(state.right[0], state.right[2], state.right[1])
        ctx2.set_source_rgb(*rgb2)
        ctx1.fill()

        if agent == 0 and state.target == 1:
            surf1, surf2 = surf2, surf1

        name1 = "%d_a.png" % self.image_counter
        name2 = "%d_b.png" % self.image_counter

        surf1.write_to_png(os.path.join(loc, name1))
        surf2.write_to_png(os.path.join(loc, name2))

        return name1, name2

    def pp(self, indices):
        return " ".join([self.reverse_vocab[i] for i in indices])
