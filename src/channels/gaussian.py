import tensorflow as tf

class GaussianChannel(object):
    def build(self, config):
        self.std = config.channel.std
        self.normalize = config.channel.normalize

    def transmit(self, msg):
        noised = msg + tf.random_normal(tf.shape(msg), stddev=self.std)
        if self.normalize:
            return tf.nn.l2_normalize(noised, 1)
        else:
            return noised
