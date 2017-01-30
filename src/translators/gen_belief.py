import net

import tensorflow as tf

class GenBeliefTranslator(object):
    def build(self, task, reconst_ph, channel, model, config):
        with tf.variable_scope("belief_translator") as scope:
            t_xa_true_rs = tf.reshape(
                    reconst_ph.t_xa_true,
                    (config.trainer.n_batch_episodes, 1, task.n_features))
            t_xa = tf.concat(1, (t_xa_true_rs, reconst_ph.t_xa_noise))
            #t_xa_drop = tf.dropout(t_xa, 0.9)
            t_xa_drop = t_xa
            #t_xa_true_drop = tf.nn.dropout(reconst_ph.t_xa_true, 0.9)
            t_xa_true_drop = reconst_ph.t_xa_true

            with tf.variable_scope("model") as model_scope:
                t_mean, self.v_model = net.mlp(
                        t_xa_true_drop,
                        (config.translator.n_hidden, config.channel.n_msg))
                t_model_logprob = -tf.reduce_sum(
                        tf.square(t_mean - reconst_ph.t_z), axis=1)
                self.t_model_loss = -tf.reduce_mean(t_model_logprob)

                model_scope.reuse_variables()
                t_all_mean, _ = net.mlp(
                        t_xa_drop, (config.translator.n_hidden, config.channel.n_msg))
                t_model_raw_belief = -tf.reduce_sum(
                        tf.square(t_all_mean-tf.expand_dims(reconst_ph.t_z, 1)),
                        axis=2)
                if config.translator.normalization == "global":
                    assert False, "you probably don't want this"
                    t_model_rs_belief = tf.nn.softmax(tf.reshape(
                            t_model_raw_belief, (1, -1)))
                    self.t_model_belief = tf.reshape(
                            t_model_rs_belief,
                            (config.trainer.n_batch_episodes,
                                config.trainer.n_distractors + 1))
                    self.t_model_weights = tf.ones(
                            (config.trainer.n_batch_episodes,))
                elif config.translator.normalization == "local":
                    self.t_model_belief = tf.nn.softmax(t_model_raw_belief)
                    #self.t_model_weights = tf.nn.softmax(t_model_logprob)
                    self.t_model_logweights = t_model_logprob
                else:
                    assert False

            with tf.variable_scope("desc") as desc_scope:
                t_dist, self.v_desc = net.mlp(
                        t_xa_true_drop,
                        (config.translator.n_hidden, len(task.lexicon)))
                        #(len(task.lexicon),))
                self.t_dist = t_dist
                t_desc_logprob = -tf.nn.softmax_cross_entropy_with_logits(
                        t_dist, reconst_ph.t_l_msg)
                self.t_desc_loss = -tf.reduce_mean(t_desc_logprob)

                desc_scope.reuse_variables()

                t_msg_tile = tf.tile(
                        tf.reshape(
                            reconst_ph.t_l_msg,
                            (config.trainer.n_batch_episodes, 1,
                                len(task.lexicon))),
                        (1, config.trainer.n_distractors + 1, 1))
                t_all_dist, _ = net.mlp(
                        t_xa_drop,
                        (config.translator.n_hidden, len(task.lexicon)))
                        #(len(task.lexicon),))
                t_all_scores = -tf.nn.softmax_cross_entropy_with_logits(
                        t_all_dist, t_msg_tile)

                if config.translator.normalization == "global":
                    t_desc_belief_raw = tf.nn.softmax(tf.reshape(
                            t_all_scores, (1, -1)))
                    t_desc_belief = tf.reshape(
                            t_desc_belief_raw,
                            (config.trainer.n_batch_episodes,
                                config.trainer.n_distractors + 1))
                    t_desc_belief_norm = (t_desc_belief /
                            tf.reduce_max(t_desc_belief))
                    self.t_desc_belief = t_desc_belief_norm
                    self.t_desc_weights = tf.ones(
                            (config.trainer.n_batch_episodes,))
                elif config.translator.normalization == "local":
                    self.t_desc_belief = tf.nn.softmax(t_all_scores)
                    #self.t_desc_weights = tf.nn.softmax(t_desc_logprob)
                    self.t_desc_logweights = t_desc_logprob
                else:
                    assert False

            optimizer = tf.train.AdamOptimizer(config.translator.step_size)

            #varz = self.v_model + self.v_desc
            #self.t_loss = self.t_desc_loss + self.t_model_loss
            #self.t_train_op = optimizer.minimize(
            #        self.t_loss, var_list=varz)
            self.t_train_model_op = optimizer.minimize(
                    self.t_model_loss, var_list=self.v_model)
            self.t_train_desc_op = optimizer.minimize(
                    self.t_desc_loss, var_list=self.v_desc)
