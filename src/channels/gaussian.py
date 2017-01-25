import tensorflow as tf

class GaussianChannel(object):
    def build(self, config):
        self.std = config.channel.std
        self.normalize = config.channel.normalize

    def transmit(self, msg):
        if self.normalize:
            msg = tf.nn.l2_normalize(msg, 1)
        return msg + tf.random_normal(tf.shape(msg), stddev=self.std)
