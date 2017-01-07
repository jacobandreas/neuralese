from comm_cell import CommCell
import net

import tensorflow as tf

class RecurrentQModel(object):
    def build(self, task, rollout_ph, replay_ph, channel, config):
        cell = CommCell(
                task.n_agents, config.model.n_hidden, config.channel.n_msg,
                task.n_actions, channel, communicate=config.model.communicate,
                symmetric=task.symmetric)

        with tf.variable_scope("net") as scope:
            tt_replay_states, _ = tf.nn.dynamic_rnn(
                    cell, replay_ph.t_x, dtype=tf.float32, scope=scope,
                    initial_state=(replay_ph.t_h, replay_ph.t_z, replay_ph.t_q))
            tt_replay_h, tt_replay_z, tt_replay_q = tt_replay_states

            scope.reuse_variables()

            tt_rollout_states, _ = cell(
                    rollout_ph.t_x,
                    (rollout_ph.t_h, rollout_ph.t_z, rollout_ph.t_q))
            tt_rollout_h, tt_rollout_z, tt_rollout_q = tt_rollout_states

            v_net = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

        with tf.variable_scope("net_next") as scope:
            tt_replay_states_next, _ = tf.nn.dynamic_rnn(
                    cell, replay_ph.t_x_next, dtype=tf.float32, scope=scope,
                    initial_state=(replay_ph.t_h_next, replay_ph.t_z_next, 
                        replay_ph.t_q_next))
            tt_replay_h_next, tt_replay_z_next, tt_replay_q_next = \
                    tt_replay_states_next

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
        self.tt_rollout_z = tt_rollout_z
        self.tt_rollout_q = tt_rollout_q
        self.t_loss = t_loss
        self.t_train_op = optimizer.minimize(t_loss, var_list=v_net)
        self.oo_update_target = [
                vn.assign(v) for v, vn in zip(v_net, v_net_next)]
