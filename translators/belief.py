import net

import tensorflow as tf

class BeliefTranslator(object):
    def __init__(self, task, base_model, n_batch, n_hidden, n_code):
        with tf.variable_scope("belief_translator"):
            t_in_sentence = tf.placeholder(tf.int32, (n_batch, task.max_sentence_len))
            t_ze = t_in_sentence[:, 0]
            t_ze_embed, v_ze_embed = net.embed(t_ze, len(task.vocab), n_code)

            t_zf = base_model.t_code

            t_xa_good_candidate = tf.concat(
                    1, (base_model.t_in_target, base_model.t_in_distractor))
            t_xa_bad_candidate = tf.concat(
                    1, (base_model.t_in_distractor, base_model.t_in_target))
            t_xa_candidates = tf.stack(
                    (t_xa_good_candidate, t_xa_bad_candidate), 1)
            t_xa_true = tf.zeros((n_batch,), tf.int32)

            t_ze_tile = tf.tile(
                    tf.reshape(t_ze_embed, (n_batch, 1, n_code)), 
                    (1, task.n_candidates, 1))
            t_zf_tile = tf.tile(
                    tf.reshape(t_zf, (n_batch, 1, n_code)), 
                    (1, task.n_candidates, 1))
            t_ze_feats = tf.concat(2, (t_ze_tile, t_xa_candidates))
            t_zf_feats = tf.concat(2, (t_zf_tile, t_xa_candidates))

            with tf.variable_scope("ze_score"):
                t_ze_scores, v_ze_scores = net.mlp(t_ze_feats, (n_hidden, 1,))
                t_ze_scores = tf.reshape(t_ze_scores, (n_batch, task.n_candidates))

            with tf.variable_scope("zf_score"):
                t_zf_scores, v_zf_scores = net.mlp(t_zf_feats, (n_hidden, 1,))
                t_zf_scores = tf.reshape(t_ze_scores, (n_batch, task.n_candidates))

            t_ze_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        t_ze_scores, t_xa_true))

            t_zf_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        t_zf_scores, t_xa_true))

            #t_loss = t_ze_loss + t_zf_loss
            t_loss = t_ze_loss

            varz = v_ze_embed + v_ze_scores + v_zf_scores

            optimizer = tf.train.AdamOptimizer(0.0003)
            t_train_op = optimizer.minimize(t_loss, var_list=varz)

            self.t_in_sentence = t_in_sentence
            self.t_loss = t_loss
            self.t_train_op = t_train_op
