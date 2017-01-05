import net

import tensorflow as tf
import numpy as np

class MlpModel(object):
    def __init__(self, task, n_batch, n_hidden, n_code):
        t_in_target = tf.placeholder(tf.float32, (n_batch, task.n_features))
        t_in_distractor = tf.placeholder(tf.float32, (n_batch, task.n_features))
        t_in_left = tf.placeholder(tf.float32, (n_batch, task.n_features))
        t_in_right = tf.placeholder(tf.float32, (n_batch, task.n_features))
        t_label = tf.placeholder(tf.int32, (n_batch,))

        with tf.variable_scope("speaker"):
            t_speaker_input = tf.concat(1, (t_in_target, t_in_distractor))
            t_code, v_speaker = net.mlp(t_speaker_input, (n_hidden, n_code), final_nonlinearity=True)
            #t_code = tf.nn.l2_normalize(t_code, 1)
            t_norm = tf.reduce_sum(tf.square(t_code), reduction_indices=(1,))
            t_norm_err = tf.maximum(0., t_norm - 1)

        t_transmitted_code = t_code + tf.random_normal(t_code.get_shape(),
                stddev=0.5)
        #t_transmitted_code = t_code

        with tf.variable_scope("listener"):
            t_listener_input = tf.concat(1, (t_in_left, t_in_right, t_transmitted_code))
            t_guess, v_listener = net.mlp(t_listener_input, (n_hidden, 2))

        t_loss = (
                tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        t_guess, t_label)
                    + t_norm_err
                ))

        t_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(t_guess, t_label, 1), tf.float32))

        optimizer = tf.train.AdamOptimizer(0.0003)
        t_train_op = optimizer.minimize(t_loss, var_list=v_speaker+v_listener)

        self.t_in_target = t_in_target
        self.t_in_distractor = t_in_distractor
        self.t_in_left = t_in_left
        self.t_in_right = t_in_right
        self.t_label = t_label
        self.t_loss = t_loss
        self.t_acc = t_acc
        self.t_train_op = t_train_op
        self.t_code = t_code
