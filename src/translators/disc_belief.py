import net

import tensorflow as tf

class DiscBeliefTranslator(object):
    def build(self, task, reconst_ph, channel, model, config):
        with tf.variable_scope("belief_translator") as scope:
            t_xb_rs = tf.reshape(
                    reconst_ph.t_xb,
                    (config.trainer.n_batch_episodes, 1, task.n_features))
            t_xa_true_rs = tf.reshape(
                    reconst_ph.t_xa_true,
                    (config.trainer.n_batch_episodes, 1, task.n_features))

            t_xb_tile = tf.tile(
                    t_xb_rs, (1, config.trainer.n_distractors + 1, 1))

            t_xa = tf.concat(1, (t_xa_true_rs, reconst_ph.t_xa_noise))

            def build_scorer(t_code):
                t_code_rs = tf.reshape(
                        t_code,
                        (config.trainer.n_batch_episodes, 1, 
                            config.channel.n_msg))
                t_code_tile = tf.tile(
                        t_code_rs, (1, config.trainer.n_distractors + 1, 1))
                t_features = tf.concat(2, (t_xa, t_xb_tile, t_code_tile))
                t_score, v_net = net.mlp(t_features, (config.model.n_hidden, 1))
                t_score_rs = tf.reshape(
                        t_score,
                        (config.trainer.n_batch_episodes,
                            config.trainer.n_distractors + 1, 1))
                t_score_sq = tf.squeeze(t_score_rs)
                t_belief = tf.nn.softmax(t_score_sq)
                t_errs = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        t_score_sq, tf.ones(
                            (config.trainer.n_batch_episodes,), tf.int32))
                t_loss = tf.reduce_mean(t_errs)
                return t_loss, t_belief, v_net

            with tf.variable_scope("model"):
                self.t_model_loss, self.t_model_belief, v_model_net = build_scorer(
                        reconst_ph.t_z)

            with tf.variable_scope("desc"):
                t_desc_embed, v_desc_embed = net.embed(
                        reconst_ph.t_desc, task.n_vocab, config.channel.n_msg)
                t_desc_pool = tf.reduce_mean(t_desc_embed, axis=1)
                self.t_desc_loss, self.t_desc_belief, v_desc_net = \
                        build_scorer(t_desc_pool)

            optimizer = tf.train.AdamOptimizer(config.model.step_size)

            varz = v_model_net + v_desc_embed + v_desc_net

            self.t_loss = self.t_desc_loss + self.t_model_loss
            self.t_train_op = optimizer.minimize(
                    self.t_loss, var_list=varz)
