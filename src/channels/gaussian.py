import tensorflow as tf

class GaussianChannel(object):
    def build(self, config):
        self.std = config.channel.std

    def transmit(self, msg):
        #return msg + tf.random_normal(tf.shape(msg), stddev=self.std)
        return msg
