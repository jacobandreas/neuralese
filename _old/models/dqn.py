import experience
import net

import tensorflow as tf
import numpy as np

DISCOUNT = 0.9

class CommCell(tf.nn.rnn_cell.RNNCell):
    def __init__(
            self, n_agents, n_hidden, n_comm, n_out, channel, communicate,
            symmetric):
        self.n_agents = n_agents
        self.n_hidden = n_hidden
        self.n_comm = n_comm
        self.n_out = n_out if isinstance(n_out, tuple) else (n_out,) * n_agents
        self.communicate = communicate
        self.symmetric = symmetric
        self.channel = channel

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
        return ((self.n_hidden,) * self.n_agents, 
                (self.n_comm,) * self.n_agents,
                self.n_out)
        #return (self.n_hidden, self.n_hidden, self.n_comm, self.n_comm,
        #        self.n_out, self.n_out)

    @property
    def output_size(self):
        return self.state_size

    def __call__(self, inputs, state, scope=None):
        assert len(inputs) == self.n_agents
        assert len(state) == 3, "len(state) is %d" % len(state)
        assert all(len(s) == self.n_agents for s in state)
        states, comms, _ = state

        with tf.variable_scope(scope or "comm_cell"):
            features = []
            if self.communicate:
                base_cell = tf.nn.rnn_cell.GRUCell(self.n_hidden)
                for i_agent in range(self.n_agents):
                    other_comms = tuple(
                            comms[_ia] for _ia in range(self.n_agents)
                            if _ia != i_agent)
                    features.append(
                            tf.concat(1, (inputs[i_agent],) + other_comms))
            else:
                # for fair comparison, let non-communicating agents use the
                # extra channel capacity for themselves
                #base_cell = tf.nn.rnn_cell.GRUCell(
                #        self.n_hidden + self.n_comm * (self.n_agents - 1))
                base_cell = tf.nn.rnn_cell.GRUCell(self.n_hidden)
                for i_agent in range(self.n_agents):
                    features.append(inputs[i_agent])

            next_states = []
            next_comms = []
            next_outs = []
            for i_agent in range(self.n_agents):
                with tf.variable_scope(self.agent_scopes[i_agent],
                        reuse=(i_agent > 0 and self.reuse_agent_scope)):
                    hidden, _ = net.mlp(
                            features[i_agent], (self.n_hidden,),
                            final_nonlinearity=True)
                    _, next_state = base_cell(hidden, states[i_agent])
                    with tf.variable_scope("comm"):
                        next_comm, _ = net.mlp(next_state, (self.n_comm,))
                    with tf.variable_scope("out"):
                        next_out, _ = net.mlp(
                                next_state, (self.n_out[i_agent],))
                    next_states.append(next_state)
                    next_comms.append(next_comm)
                    next_outs.append(next_out)

            #transmitted = [self.channel.transmit(c) for c in next_comms]
            transmitted = next_comms

        next_state = (tuple(next_states), tuple(transmitted), tuple(next_outs))
        #next_state = tuple(next_states) + tuple(transmitted) + tuple(next_outs)
        return next_state, next_state

#class CommCell(tf.nn.rnn_cell.RNNCell):
#    def __init__(self, n_hidden, n_code, n_output, communicate=True):
#        self.n_hidden = n_hidden
#        self.n_output = n_output
#        self.n_code = n_code
#        self.communicate = communicate
#
#    @property
#    def state_size(self):
#        return (self.n_hidden, self.n_hidden, self.n_code, self.n_code)
#
#    @property
#    def output_size(self):
#        return self.state_size
#
#    def __call__(self, inputs, state, scope=None):
#        input_a, input_b = inputs
#        state_a, state_b, comm_a, comm_b = state
#        with tf.variable_scope(scope or "comm_cell") as scope:
#            base_cell = tf.nn.rnn_cell.GRUCell(self.n_hidden)
#            with tf.variable_scope("agent") as state_scope:
#                if self.communicate:
#                    features_a = tf.concat(1, (input_a, comm_b))
#                    features_b = tf.concat(1, (input_b, comm_a))
#                else:
#                    features_a = input_a
#                    features_b = input_b
#
#                hidden_a, _ = net.mlp(features_a, (self.n_hidden,),
#                        final_nonlinearity=True)
#                next_out_a, next_state_a = base_cell(hidden_a, state_a)
#                state_scope.reuse_variables()
#                hidden_b, _ = net.mlp(features_b, (self.n_hidden,),
#                        final_nonlinearity=True)
#                next_out_b, next_state_b = base_cell(hidden_b, state_b)
#
#            with tf.variable_scope("agent/comm") as comm_scope:
#                next_comm_a, _ = net.mlp(next_state_a, (self.n_code,))
#                comm_scope.reuse_variables()
#                next_comm_b, _ = net.mlp(next_state_b, (self.n_code,))
#
#            next_comm_a = next_comm_a + tf.random_normal(
#                    tf.shape(next_comm_a), stddev=0.5)
#            next_comm_b = next_comm_b + tf.random_normal(
#                    tf.shape(next_comm_b), stddev=0.5)
#
#            next_state = (next_state_a, next_state_b, next_comm_a, next_comm_b)
#
#            return next_state, next_state

class DqnModel(object):
    def __init__(self, task, n_batch, n_history, n_hidden, n_code, communicate):
        roph = experience.RolloutPlaceholders(task, n_hidden, n_code)
        reph = experience.ReplayPlaceholders(task, n_hidden, n_code, n_batch, n_history)
        #t_act_state_a = tf.placeholder(tf.float32, (1, task.n_features))
        #t_act_state_b = tf.placeholder(tf.float32, (1, task.n_features))
        #t_act_hidden_a = tf.placeholder(tf.float32, (1, n_hidden))
        #t_act_hidden_b = tf.placeholder(tf.float32, (1, n_hidden))
        #t_act_comm_a = tf.placeholder(tf.float32, (1, n_code))
        #t_act_comm_b = tf.placeholder(tf.float32, (1, n_code))
        #t_init_mstate1 = (
        #        (tf.placeholder(tf.float32, (None, n_hidden)),
        #         tf.placeholder(tf.float32, (None, n_hidden))),
        #        (tf.placeholder(tf.float32, (None, n_code)),
        #         tf.placeholder(tf.float32, (None, n_code))),
        #        (tf.placeholder(tf.float32, (None, task.n_actions)),
        #         tf.placeholder(tf.float32, (None, task.n_actions))))
        #t_init_mstate2 = (
        #        (tf.placeholder(tf.float32, (None, n_hidden)),
        #         tf.placeholder(tf.float32, (None, n_hidden))),
        #        (tf.placeholder(tf.float32, (None, n_code)),
        #         tf.placeholder(tf.float32, (None, n_code))),
        #        (tf.placeholder(tf.float32, (None, task.n_actions)),
        #         tf.placeholder(tf.float32, (None, task.n_actions))))
        #t_state1_a = tf.placeholder(tf.float32, (n_batch, n_history, task.n_features))
        #t_state1_b = tf.placeholder(tf.float32, (n_batch, n_history, task.n_features))
        #t_state2_a = tf.placeholder(tf.float32, (n_batch, n_history, task.n_features))
        #t_state2_b = tf.placeholder(tf.float32, (n_batch, n_history, task.n_features))
        #t_action_a = tf.placeholder(tf.float32, (n_batch, n_history, task.n_actions))
        #t_action_b = tf.placeholder(tf.float32, (n_batch, n_history, task.n_actions))
        #t_reward = tf.placeholder(tf.float32, (n_batch, n_history))
        #t_terminal = tf.placeholder(tf.float32, (n_batch, n_history))
        #t_mask = tf.placeholder(tf.float32, (n_batch, n_history))

        cell = CommCell(2, n_hidden, n_code, (task.n_actions, task.n_actions),
                None, communicate, True)

        with tf.variable_scope("net") as scope:
            states, _ = tf.nn.dynamic_rnn(
                    cell, reph.t_x, dtype=tf.float32,
                    scope=scope, initial_state=(reph.t_h, reph.t_z, reph.t_q))
            #with tf.variable_scope("agent/out") as pscope:
            #    t_q_a, _ = net.mlp(states[0], (task.n_actions,))
            #    pscope.reuse_variables()
            #    t_q_b, _ = net.mlp(states[1], (task.n_actions,))
            t_q_a, t_q_b = states[-1]
            scope.reuse_variables()

            act_states, _ = cell(
                    roph.t_x,
                    (roph.t_h, roph.t_z, roph.t_q))
            #with tf.variable_scope("agent/out") as pscope:
            #    t_act_q_a, _ = net.mlp(act_states[0], (task.n_actions,))
            #    pscope.reuse_variables()
            #    t_act_q_b, _ = net.mlp(act_states[1], (task.n_actions,))
            ((t_act_next_hidden_a, t_act_next_hidden_b), (t_act_next_comm_a,
                    t_act_next_comm_b), (t_act_q_a, t_act_q_b)) = act_states

            v = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

        with tf.variable_scope("net_target") as scope:
            states, _ = tf.nn.dynamic_rnn(
                    cell, reph.t_x_next, dtype=tf.float32,
                    scope=scope, initial_state=(reph.t_h_next, reph.t_z_next,
                        reph.t_q_next))
            #with tf.variable_scope("agent/out") as pscope:
            #    t_q_a_target, _ = net.mlp(states[0], (task.n_actions,))
            #    pscope.reuse_variables()
            #    t_q_b_target, _ = net.mlp(states[1], (task.n_actions,))
            t_q_a_target, t_q_b_target = states[-1]
            v_target = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

        self.t_act_a = tf.argmax(t_act_q_a, axis=1)
        self.t_act_b = tf.argmax(t_act_q_b, axis=1)
        self.t_act_q_a = t_act_q_a
        self.t_act_q_b = t_act_q_b

        t_td_a = (
            DISCOUNT * tf.reduce_max(t_q_a_target, axis=2) * (1 - reph.t_terminal)
            + reph.t_reward
            - tf.reduce_sum(t_q_a * reph.t_action[0], axis=2))

        t_td_b = (
            DISCOUNT * tf.reduce_max(t_q_b_target, axis=2) * (1 - reph.t_terminal)
            + reph.t_reward
            - tf.reduce_sum(t_q_b * reph.t_action[1], axis=2))

        self.t_loss = (
                tf.reduce_mean(reph.t_mask * tf.square(t_td_a))
                + tf.reduce_mean(reph.t_mask * tf.square(t_td_b)))

        optimizer = tf.train.AdamOptimizer(0.0003)

        self.t_train_op = optimizer.minimize(self.t_loss, var_list=v)
        self.t_update_target_ops = [_vt.assign(_v) for _v, _vt in zip(v, v_target)]

        self.t_act_state_a = roph.t_x[0]
        self.t_act_state_b = roph.t_x[1]
        self.t_state1_a = reph.t_x[0]
        self.t_state1_b = reph.t_x[1]
        self.t_state2_a = reph.t_x_next[0]
        self.t_state2_b = reph.t_x_next[1]
        self.t_init_mstate1 = (reph.t_h[0], reph.t_h[1], reph.t_z[0], reph.t_z[1])
        self.t_init_mstate2 = (reph.t_h_next[0], reph.t_h_next[1],
                reph.t_z_next[0], reph.t_z_next[1])
        self.t_action_a = reph.t_action[0]
        self.t_action_b = reph.t_action[1]

        self.t_reward = reph.t_reward
        self.t_terminal = reph.t_terminal
        self.t_mask = reph.t_mask

        self.v = v

        self.t_act_hidden_a = roph.t_h[0]
        self.t_act_hidden_b = roph.t_h[1]
        self.t_act_comm_a = roph.t_z[0]
        self.t_act_comm_b = roph.t_z[1]
        self.t_act_next_hidden_a = t_act_next_hidden_a
        self.t_act_next_hidden_b = t_act_next_hidden_b
        self.t_act_next_comm_a = t_act_next_comm_a
        self.t_act_next_comm_b = t_act_next_comm_b
