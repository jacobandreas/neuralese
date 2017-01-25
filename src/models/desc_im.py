import net

import tensorflow as tf

def embed_all(inputs, size):
    out = []
    with tf.variable_scope("embed_all") as scope:
        for inp in inputs:
            #t_emb, _ = net.embed(inp, count, size)
            #t_pool = tf.reduce_mean(t_emb, axis=-2)
            #out.append(t_pool)
            #scope.reuse_variables()
            t_emb, _ = net.mlp(inp, (size,))
            out.append(t_emb)
            scope.reuse_variables()
    return out

class DescCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, n_agents, n_lex, n_hidden, n_out, symmetric):
        self.n_lex = n_lex
        self.n_agents = n_agents
        self.n_hidden = n_hidden
        self.n_out = n_out if isinstance(n_out, tuple) else (n_out,) * n_agents
        self.symmetric = symmetric

        self.agent_scopes = []
        if symmetric:
            for i_agent in range(n_agents):
                self.agent_scopes.append("agent")
            self.reuse_agent_scope = True
        else:
            for i_agent in range(n_agents):
                self.agent_scopes.append("agent_%d" % i_agent)
            self.reuse_agent_scope = False

    @property
    def state_size(self):
        return ((self.n_hidden,) * self.n_agents, self.n_out,
                (self.n_lex,) * self.n_agents)

    @property
    def output_size(self):
        return self.state_size

    def __call__(self, inputs, state, scope=None):
        assert len(inputs) == self.n_agents
        assert len(state) == 3
        assert all(len(s) == self.n_agents for s in state)

        base_cell = tf.nn.rnn_cell.GRUCell(self.n_hidden)
        states, _, _ = state

        with tf.variable_scope(scope or "desc_cell"):
            next_states = []
            next_outs = []
            next_l_msgs = []
            for i_agent in range(self.n_agents):
                with tf.variable_scope(self.agent_scopes[i_agent],
                        reuse=(i_agent > 0 and self.reuse_agent_scope)):
                    _, next_state = base_cell(inputs[i_agent], states[i_agent])
                    with tf.variable_scope("out"):
                        next_out, _ = net.mlp(
                                next_state, (self.n_out[i_agent],))
                    with tf.variable_scope("gen"):
                        next_l_msg, _ = net.mlp(
                                next_state, (self.n_lex,))
                    next_states.append(next_state)
                    next_outs.append(next_out)
                    next_l_msgs.append(next_l_msg)

        next_state = (tuple(next_states), tuple(next_outs), tuple(next_l_msgs))
        return next_state, next_state

class DescriptionImitationModel(object):
    def build(self, task, rollout_ph, replay_ph, channel, config):
        cell = DescCell(
                task.n_agents, len(task.lexicon), config.model.n_hidden,
                task.n_actions, task.symmetric)

        replay_x = replay_ph.t_x
        replay_x_next = replay_ph.t_x

        with tf.variable_scope("desc_net") as scope:
            tt_embeddings = embed_all(
                    replay_ph.t_l_msg, config.model.n_embed)
            tt_features = tuple(
                    tf.concat(2, (e, x)) for e, x
                    in zip(tt_embeddings, replay_x))
            tt_replay_states, _ = tf.nn.dynamic_rnn(
                    cell, tt_features, dtype=tf.float32, scope=scope,
                    initial_state=(replay_ph.t_l_h, replay_ph.t_fake_q,
                        replay_ph.t_fake_l_msg))
            tt_replay_h, tt_replay_score, tt_replay_gen = tt_replay_states

            scope.reuse_variables()

            tt_rollout_embeddings = embed_all(
                    rollout_ph.t_l_msg, config.model.n_embed)
            tt_rollout_features = tuple(
                    tf.concat(1, (e, x)) for e, x
                    in zip(tt_rollout_embeddings, rollout_ph.t_x))
            tt_rollout_states, _ = cell(
                    tt_rollout_features,
                    (rollout_ph.t_l_h, rollout_ph.t_fake_q,
                        rollout_ph.t_fake_l_msg))
            tt_rollout_h, tt_rollout_score, tt_rollout_l_msg = tt_rollout_states

            v_net = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

        tt_loss = []
        for t_score, t_action in zip(tt_replay_score, replay_ph.t_action_index):
            tt_loss.append(tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        t_score, t_action)))

        for t_gen, t_gen_target in zip(tt_replay_gen, replay_ph.t_l_msg_target):
            tt_loss.append(tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        t_gen, t_gen_target)))

        t_loss = tf.reduce_sum(tt_loss)

        optimizer = tf.train.AdamOptimizer(config.model.step_size)

        self.zero_state = cell.zero_state
        self.t_loss = t_loss
        self.tt_rollout_h = tt_rollout_h
        self.tt_rollout_q = tt_rollout_score
        self.tt_rollout_l_msg = tt_rollout_l_msg

        self.t_train_op = optimizer.minimize(t_loss, var_list=v_net)
