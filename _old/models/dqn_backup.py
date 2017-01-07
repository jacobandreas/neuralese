import net

import tensorflow as tf
import numpy as np

DISCOUNT = 0.9

class CommCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, n_hidden, n_code, n_output, communicate=True):
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_code = n_code
        self.communicate = communicate

    @property
    def state_size(self):
        return (self.n_hidden, self.n_hidden, self.n_code, self.n_code)

    @property
    def output_size(self):
        return self.state_size

    def __call__(self, inputs, state, scope=None):
        input_a, input_b = inputs
        state_a, state_b, comm_a, comm_b = state
        with tf.variable_scope(scope or "comm_cell") as scope:
            base_cell = tf.nn.rnn_cell.GRUCell(self.n_hidden)
            with tf.variable_scope("agent") as state_scope:
                if self.communicate:
                    features_a = tf.concat(1, (input_a, comm_b))
                    features_b = tf.concat(1, (input_b, comm_a))
                else:
                    features_a = input_a
                    features_b = input_b

                hidden_a, _ = net.mlp(features_a, (self.n_hidden,),
                        final_nonlinearity=True)
                next_out_a, next_state_a = base_cell(hidden_a, state_a)
                state_scope.reuse_variables()
                hidden_b, _ = net.mlp(features_b, (self.n_hidden,),
                        final_nonlinearity=True)
                next_out_b, next_state_b = base_cell(hidden_b, state_b)

                with tf.variable_scope("comm") as comm_scope:
                    next_comm_a, _ = net.mlp(next_state_a, (self.n_code,))
                    comm_scope.reuse_variables()
                    next_comm_b, _ = net.mlp(next_state_b, (self.n_code,))

            next_comm_a = next_comm_a + tf.random_normal(
                    tf.shape(next_comm_a), stddev=0.5)
            next_comm_b = next_comm_b + tf.random_normal(
                    tf.shape(next_comm_b), stddev=0.5)

            next_state = (next_state_a, next_state_b, next_comm_a, next_comm_b)

            return next_state, next_state

class DqnModel(object):
    def __init__(self, task, n_batch, n_history, n_hidden, n_code, communicate):
        t_act_state_a = tf.placeholder(tf.float32, (1, task.n_features))
        t_act_state_b = tf.placeholder(tf.float32, (1, task.n_features))
        t_act_hidden_a = tf.placeholder(tf.float32, (1, n_hidden))
        t_act_hidden_b = tf.placeholder(tf.float32, (1, n_hidden))
        t_act_comm_a = tf.placeholder(tf.float32, (1, n_code))
        t_act_comm_b = tf.placeholder(tf.float32, (1, n_code))
        t_init_mstate1 = (
                tf.placeholder(tf.float32, (None, n_hidden)),
                tf.placeholder(tf.float32, (None, n_hidden)),
                tf.placeholder(tf.float32, (None, n_code)),
                tf.placeholder(tf.float32, (None, n_code)))
        t_init_mstate2 = (
                tf.placeholder(tf.float32, (None, n_hidden)),
                tf.placeholder(tf.float32, (None, n_hidden)),
                tf.placeholder(tf.float32, (None, n_code)),
                tf.placeholder(tf.float32, (None, n_code)))
        t_state1_a = tf.placeholder(tf.float32, (n_batch, n_history, task.n_features))
        t_state1_b = tf.placeholder(tf.float32, (n_batch, n_history, task.n_features))
        t_state2_a = tf.placeholder(tf.float32, (n_batch, n_history, task.n_features))
        t_state2_b = tf.placeholder(tf.float32, (n_batch, n_history, task.n_features))
        t_action_a = tf.placeholder(tf.float32, (n_batch, n_history, task.n_actions))
        t_action_b = tf.placeholder(tf.float32, (n_batch, n_history, task.n_actions))
        t_reward = tf.placeholder(tf.float32, (n_batch, n_history))
        t_terminal = tf.placeholder(tf.float32, (n_batch, n_history))
        t_mask = tf.placeholder(tf.float32, (n_batch, n_history))

        with tf.variable_scope("net") as scope:
            cell = CommCell(n_hidden, n_code, task.n_actions, communicate)
            states, _ = tf.nn.dynamic_rnn(
                    cell, (t_state1_a, t_state1_b), dtype=tf.float32,
                    scope=scope, initial_state=t_init_mstate1)
            with tf.variable_scope("agent/out") as pscope:
                t_q_a, _ = net.mlp(states[0], (task.n_actions,))
                pscope.reuse_variables()
                t_q_b, _ = net.mlp(states[1], (task.n_actions,))
            scope.reuse_variables()

            act_states, _ = cell(
                    (t_act_state_a, t_act_state_b),
                    (t_act_hidden_a, t_act_hidden_b, t_act_comm_a, t_act_comm_b))
            with tf.variable_scope("agent/out") as pscope:
                t_act_q_a, _ = net.mlp(act_states[0], (task.n_actions,))
                pscope.reuse_variables()
                t_act_q_b, _ = net.mlp(act_states[1], (task.n_actions,))
            (t_act_next_hidden_a, t_act_next_hidden_b, t_act_next_comm_a,
                    t_act_next_comm_b) = act_states

            v = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

        with tf.variable_scope("net_target") as scope:
            cell = CommCell(n_hidden, n_code, task.n_actions, communicate)
            states, _ = tf.nn.dynamic_rnn(
                    cell, (t_state2_a, t_state2_b), dtype=tf.float32,
                    scope=scope, initial_state=t_init_mstate2)
            with tf.variable_scope("agent/out") as pscope:
                t_q_a_target, _ = net.mlp(states[0], (task.n_actions,))
                pscope.reuse_variables()
                t_q_b_target, _ = net.mlp(states[1], (task.n_actions,))
            v_target = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

        self.t_act_a = tf.argmax(t_act_q_a, axis=1)
        self.t_act_b = tf.argmax(t_act_q_b, axis=1)
        self.t_act_q_a = t_act_q_a

        t_td_a = (
            DISCOUNT * tf.reduce_max(t_q_a_target, axis=2) * (1 - t_terminal)
            + t_reward
            - tf.reduce_sum(t_q_a * t_action_a, axis=2))

        t_td_b = (
            DISCOUNT * tf.reduce_max(t_q_b_target, axis=2) * (1 - t_terminal)
            + t_reward
            - tf.reduce_sum(t_q_b * t_action_b, axis=2))

        self.t_loss = (
                tf.reduce_mean(t_mask * tf.square(t_td_a))
                + tf.reduce_mean(t_mask * tf.square(t_td_b)))

        optimizer = tf.train.AdamOptimizer(0.0003)

        self.t_train_op = optimizer.minimize(self.t_loss, var_list=v)
        self.t_update_target_ops = [_vt.assign(_v) for _v, _vt in zip(v, v_target)]

        self.t_act_state_a = t_act_state_a
        self.t_act_state_b = t_act_state_b
        self.t_state1_a = t_state1_a
        self.t_state1_b = t_state1_b
        self.t_state2_a = t_state2_a
        self.t_state2_b = t_state2_b
        self.t_init_mstate1 = t_init_mstate1
        self.t_init_mstate2 = t_init_mstate2
        self.t_action_a = t_action_a
        self.t_action_b = t_action_b

        self.t_reward = t_reward
        self.t_terminal = t_terminal
        self.t_mask = t_mask

        self.v = v

        self.t_act_hidden_a = t_act_hidden_a
        self.t_act_hidden_b = t_act_hidden_b
        self.t_act_comm_a = t_act_comm_a
        self.t_act_comm_b = t_act_comm_b
        self.t_act_next_hidden_a = t_act_next_hidden_a
        self.t_act_next_hidden_b = t_act_next_hidden_b
        self.t_act_next_comm_a = t_act_next_comm_a
        self.t_act_next_comm_b = t_act_next_comm_b
