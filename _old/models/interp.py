import net

import tensorflow as tf

class InterpreterModel(object):
    def __init__(self, task, base_model, n_batch, n_code):
        t_in_sentence = tf.placeholder(tf.int32, (n_batch, task.max_sentence_len))
        t_embed, v_embed = net.embed(t_in_sentence, len(task.vocab), n_code)
        t_raw_repr = tf.reduce_sum(t_embed, reduction_indices=(1,))
        t_repr = tf.nn.l2_normalize(t_raw_repr, 1)
        t_energy = 1-tf.reduce_sum(
                t_repr * base_model.t_code, reduction_indices=(1,))
        t_loss = tf.reduce_mean(t_energy)

        optimizer = tf.train.AdamOptimizer(0.0003)
        t_train_op = optimizer.minimize(t_loss, var_list=v_embed)

        self.t_in_sentence = t_in_sentence
        self.t_embedding = v_embed[0]
        self.t_repr = t_repr
        self.t_loss = t_loss
        self.t_train_op = t_train_op
