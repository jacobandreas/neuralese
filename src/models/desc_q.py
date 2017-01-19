import net

import tensorflow as tf

def embed_all(inputs, count, size):
    out = []
    with tf.variable_scope("embed_all") as scope:
        for inp in inputs:
            t_emb, _ = net.embed(inp, count, size)
            t_pool = tf.reduce_mean(t_emb, axis=-2)
            out.append(t_pool)
            scope.reuse_variables()
    return out

class DescCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, n_agents, n_vocab, n_hidden, n_out, symmetric):
        self.n_vocab = n_vocab
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
        return ((self.n_hidden,) * self.n_agents, self.n_out)

    @property
    def output_size(self):
        return self.state_size

    def __call__(self, inputs, state, scope=None):
        assert len(inputs) == self.n_agents
        assert len(state) == 2
        assert all(len(s) == self.n_agents for s in state)

        base_cell = tf.nn.rnn_cell.GRUCell(self.n_hidden)
        states, _ = state

        with tf.variable_scope(scope or "desc_cell"):
            next_states = []
            next_outs = []
            for i_agent in range(self.n_agents):
                with tf.variable_scope(self.agent_scopes[i_agent],
                        reuse=(i_agent > 0 and self.reuse_agent_scope)):
                    _, next_state = base_cell(inputs[i_agent], states[i_agent])
                    with tf.variable_scope("out"):
                        next_out, _ = net.mlp(
                                next_state, (self.n_out[i_agent],))
                    next_states.append(next_state)
                    next_outs.append(next_out)

        next_state = (tuple(next_states), tuple(next_outs))
        return next_state, next_state

class DescriptionQModel(object):
    def build(self, task, rollout_ph, replay_ph, channel, config):
        cell = DescCell(
                task.n_agents, task.n_vocab, config.model.n_hidden,
                task.n_actions, task.symmetric)

        with tf.variable_scope("desc_net") as scope:
            tt_embeddings = embed_all(
                    replay_ph.t_desc, task.n_vocab, config.model.n_embed)
            tt_features = tuple(
                    tf.concat(2, (e, x)) for e, x
                    in zip(tt_embeddings, replay_ph.t_x))
            tt_replay_states, _ = tf.nn.dynamic_rnn(
                    cell, tt_features, dtype=tf.float32, scope=scope,
                    initial_state=(replay_ph.t_dh, replay_ph.t_q))
            tt_replay_h, tt_replay_q = tt_replay_states

            scope.reuse_variables()

            tt_rollout_embeddings = embed_all(
                    rollout_ph.t_desc, task.n_vocab, config.model.n_embed)
            tt_rollout_features = tuple(
                    tf.concat(1, (e, x)) for e, x
                    in zip(tt_rollout_embeddings, rollout_ph.t_x))
            tt_rollout_states, _ = cell(
                    tt_rollout_features, (rollout_ph.t_desc_h, rollout_ph.t_q))
            tt_rollout_h, tt_rollout_q = tt_rollout_states

            v_net = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

        with tf.variable_scope("desc_net_next") as scope:
            tt_embeddings_next = embed_all(
                    replay_ph.t_desc_next, task.n_vocab, config.model.n_embed)
            tt_features_next = tuple(
                    tf.concat(2, (e, x)) for e, x
                    in zip(tt_embeddings_next, replay_ph.t_x_next))
            tt_replay_states_next, _ = tf.nn.dynamic_rnn(
                    cell, tt_features_next, dtype=tf.float32, scope=scope,
                    initial_state=(replay_ph.t_dh_next, replay_ph.t_q_next))
            tt_replay_h_next, tt_replay_q_next = tt_replay_states_next

            v_net_next = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

        tt_td = []
        tt_loss = []
        for t_q, t_q_next, t_action in zip(
                tt_replay_q, tt_replay_q_next, replay_ph.t_action):
            t_td = (
                config.model.discount * tf.reduce_max(t_q_next, axis=2)
                    * (1 - replay_ph.t_terminal)
                + replay_ph.t_reward
                - tf.reduce_sum(t_q * t_action, axis=2))
            tt_td.append(t_td)
            tt_loss.append(tf.reduce_mean(replay_ph.t_mask * tf.square(t_td)))

        t_loss = tf.reduce_sum(tt_loss)

        optimizer = tf.train.AdamOptimizer(config.model.step_size)

        self.zero_state = cell.zero_state
        self.tt_rollout_h = tt_rollout_h
        self.tt_rollout_q = tt_rollout_q
        self.t_loss = t_loss

        self.t_q = tt_replay_q
        self.t_q_next = tt_replay_q_next
        self.t_td = tt_td

        self.t_debug_1 = (config.model.discount 
                * tf.reduce_max(t_q_next, axis=2) 
                * (1 - replay_ph.t_terminal))
        self.t_debug_2 = replay_ph.t_reward
        self.t_debug_3 = tf.reduce_sum(t_q * t_action, axis=2)
        #self.t_q_n = tt_replay_q_next
        #self.t_rew = replay_ph.t_reward

        self.t_train_op = optimizer.minimize(t_loss, var_list=v_net)
        self.oo_update_target = [
                vn.assign(v) for v, vn in zip(v_net, v_net_next)]
