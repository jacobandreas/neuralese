from collections import namedtuple
import numpy as np
import tensorflow as tf

Experience = namedtuple("Experience", ["s1", "m1", "a", "s2", "m2", "r", "t"])

class RolloutPlaceholders(object):
    def __init__(self, task, config):
        t_x = []
        t_h = []
        t_z = []
        t_fake_q = []
        t_fake_l_msg = []
        t_l_msg = []
        t_l_h = []
        for i_agent in range(task.n_agents):
            t_x.append(tf.placeholder(
                    tf.float32, (config.trainer.n_rollout_episodes,
                        task.n_features)))
            t_h.append(tf.placeholder(
                    tf.float32, (config.trainer.n_rollout_episodes,
                        config.model.n_hidden)))
            t_z.append(tf.placeholder(
                    tf.float32, (config.trainer.n_rollout_episodes,
                        config.channel.n_msg)))
            t_fake_q.append(tf.zeros(
                    (config.trainer.n_rollout_episodes, task.n_actions[i_agent]),
                    tf.float32))
            t_fake_l_msg.append(tf.zeros(
                    (config.trainer.n_rollout_episodes, len(task.lexicon))))
            t_l_msg.append(tf.placeholder(
                    tf.float32, (config.trainer.n_rollout_episodes,
                        len(task.lexicon))))
            t_l_h.append(tf.placeholder(
                    tf.float32, (config.trainer.n_rollout_episodes,
                        config.model.n_hidden)))
        self.t_x = tuple(t_x)
        self.t_h = tuple(t_h)
        self.t_z = tuple(t_z)
        self.t_fake_q = tuple(t_fake_q)
        self.t_fake_l_msg = tuple(t_fake_l_msg)
        self.t_l_msg = tuple(t_l_msg)
        self.t_l_h = tuple(t_l_h)

    def feed(self, hs, zs, l_hs, worlds, task, config):
        out = {}
        obs = [w.obs() for w in worlds]
        l_msg = [np.zeros((config.trainer.n_rollout_episodes, len(task.lexicon)))
                for _ in range(task.n_agents)]
        for i, w in enumerate(worlds):
            for i_agent in range(task.n_agents):
                l_msg[i_agent][i, :] = w.l_msg[i_agent]
        for i_t, t in enumerate(self.t_x):
            out[t] = [
                    obs[i][i_t] for i in range(config.trainer.n_rollout_episodes)]
        for i_agent, t in enumerate(self.t_l_msg):
            out[t] = [
                    l_msg[i_agent][i, :] 
                    for i in range(config.trainer.n_rollout_episodes)]
        out[self.t_h] = hs
        out[self.t_z] = zs
        out[self.t_l_h] = l_hs
        return out

class ReconstructionPlaceholders(object):
    def __init__(self, task, config):
        self.t_xb = tf.placeholder(
                tf.float32,
                (config.trainer.n_batch_episodes, task.n_features))
        self.t_z = tf.placeholder(
                tf.float32,
                (config.trainer.n_batch_episodes, config.channel.n_msg))
        self.t_xa_true = tf.placeholder(
                tf.float32,
                (config.trainer.n_batch_episodes, task.n_features))
        self.t_xa_noise = tf.placeholder(
                tf.float32,
                (config.trainer.n_batch_episodes, config.trainer.n_distractors,
                    task.n_features))
        self.t_l_msg = tf.placeholder(tf.float32,
                (config.trainer.n_batch_episodes, len(task.lexicon)))

    def feed(self, experiences, obs_agent, hidden_agent, task, config):
        xb = np.zeros((config.trainer.n_batch_episodes, task.n_features))
        z = np.zeros((config.trainer.n_batch_episodes, config.channel.n_msg))
        l_msg = np.zeros(
                (config.trainer.n_batch_episodes, len(task.lexicon)),
                dtype=np.float32)
        xa_true = np.zeros((config.trainer.n_batch_episodes, task.n_features))
        xa_noise = np.zeros(
                (config.trainer.n_batch_episodes, config.trainer.n_distractors,
                    task.n_features))

        for i, experience in enumerate(experiences):
            state = experience.s1
            message = experience.m1[1][hidden_agent]
            xb[i, :] = state.obs()[obs_agent]
            z[i, :] = message
            l_msg[i, :] = state.l_msg[obs_agent]
            xa_true[i, :] = state.obs()[hidden_agent]
            distractors = task.distractors_for(
                    state, obs_agent, config.trainer.n_distractors)
            for i_dis in range(config.trainer.n_distractors):
                dis, prob = distractors[i_dis]
                xa_noise[i, i_dis, :] = dis.obs()[hidden_agent]

        return {
            self.t_xb: xb,
            self.t_z: z,
            self.t_xa_true: xa_true,
            self.t_xa_noise: xa_noise,
            self.t_l_msg: l_msg
        }

class ReplayPlaceholders(object):
    def __init__(self, task, config):
        self.t_reward = tf.placeholder(
                tf.float32,
                (config.trainer.n_batch_episodes,
                    config.trainer.n_batch_history))
        self.t_terminal = tf.placeholder(
                tf.float32,
                (config.trainer.n_batch_episodes,
                    config.trainer.n_batch_history))
        self.t_mask = tf.placeholder(
                tf.float32,
                (config.trainer.n_batch_episodes,
                    config.trainer.n_batch_history))

        t_x = []
        t_x_next = []
        t_z = []
        t_z_next = []
        t_h = []
        t_h_next = []
        t_fake_q = []
        t_fake_l_msg = []
        t_l_h = []
        t_l_h_next = []
        t_l_msg = []
        t_l_msg_next = []
        t_l_msg_target = []
        t_fakegen = []
        t_action = []
        t_action_index = []
        for i_agent in range(task.n_agents):
            t_x.append(tf.placeholder(
                    tf.float32,
                    (config.trainer.n_batch_episodes,
                        config.trainer.n_batch_history,
                        task.n_features)))
            t_x_next.append(tf.placeholder(
                    tf.float32,
                    (config.trainer.n_batch_episodes,
                        config.trainer.n_batch_history,
                        task.n_features)))

            t_l_msg.append(tf.placeholder(
                    tf.float32,
                    (config.trainer.n_batch_episodes,
                        config.trainer.n_batch_history,
                        len(task.lexicon))))
            t_l_msg_next.append(tf.placeholder(
                    tf.float32,
                    (config.trainer.n_batch_episodes,
                        config.trainer.n_batch_history,
                        len(task.lexicon))))
            t_l_msg_target.append(tf.placeholder(
                    tf.float32,
                    (config.trainer.n_batch_episodes,
                        config.trainer.n_batch_history,
                        len(task.lexicon))))

            t_z.append(tf.placeholder(
                    tf.float32,
                    (config.trainer.n_batch_episodes, config.channel.n_msg)))
            t_z_next.append(tf.placeholder(
                    tf.float32,
                    (config.trainer.n_batch_episodes, config.channel.n_msg)))

            t_h.append(tf.placeholder(
                    tf.float32,
                    (config.trainer.n_batch_episodes, config.model.n_hidden)))
            t_h_next.append(tf.placeholder(
                    tf.float32,
                    (config.trainer.n_batch_episodes, config.model.n_hidden)))

            t_l_h.append(tf.placeholder(
                    tf.float32,
                    (config.trainer.n_batch_episodes, config.model.n_hidden)))
            t_l_h_next.append(tf.placeholder(
                    tf.float32,
                    (config.trainer.n_batch_episodes, config.model.n_hidden)))

            t_fake_q.append(tf.zeros(
                    (config.trainer.n_batch_episodes, task.n_actions[i_agent]),
                    tf.float32))
            t_fake_l_msg.append(tf.zeros(
                    (config.trainer.n_batch_episodes, len(task.lexicon)),
                    tf.float32))

            t_action.append(tf.placeholder(
                    tf.float32,
                    (config.trainer.n_batch_episodes,
                        config.trainer.n_batch_history,
                        task.n_actions[i_agent])))
            t_action_index.append(tf.placeholder(
                    tf.int32,
                    (config.trainer.n_batch_episodes,
                        config.trainer.n_batch_history)))
        self.t_x = tuple(t_x)
        self.t_x_next = tuple(t_x_next)
        self.t_z = tuple(t_z)
        self.t_z_next = tuple(t_z_next)
        self.t_h = tuple(t_h)
        self.t_h_next = tuple(t_h_next)
        self.t_fake_q = tuple(t_fake_q)
        self.t_l_h = tuple(t_l_h)
        self.t_l_h_next = tuple(t_l_h_next)
        self.t_l_msg = tuple(t_l_msg)
        self.t_l_msg_next = tuple(t_l_msg_next)
        self.t_l_msg_target = tuple(t_l_msg_target)
        self.t_fake_l_msg = tuple(t_fake_l_msg)
        self.t_action = tuple(t_action)
        self.t_action_index = tuple(t_action_index)

    def feed(self, episodes, task, config):
        n_batch = len(episodes)
        n_history = config.trainer.n_batch_history

        reward = np.zeros((n_batch, n_history))
        terminal = np.zeros((n_batch, n_history))
        mask = np.zeros((n_batch, n_history))

        x = []
        x_next = []
        h = []
        h_next = []
        z = []
        z_next = []
        l_h = []
        l_h_next = []
        action = []
        action_index = []
        l_msg = []
        l_msg_next = []
        l_msg_target = [None, None]
        for i_agent in range(task.n_agents):
            x.append(np.zeros((n_batch, n_history, task.n_features)))
            x_next.append(np.zeros((n_batch, n_history, task.n_features)))
            h.append(np.zeros((n_batch, config.model.n_hidden)))
            h_next.append(np.zeros((n_batch, config.model.n_hidden)))
            z.append(np.zeros((n_batch, config.channel.n_msg)))
            z_next.append(np.zeros((n_batch, config.channel.n_msg)))
            l_h.append(np.zeros((n_batch, config.model.n_hidden)))
            l_h_next.append(np.zeros((n_batch, config.model.n_hidden)))
            action.append(
                    np.zeros((n_batch, n_history, task.n_actions[i_agent])))
            action_index.append(np.zeros((n_batch, n_history)))
            l_msg.append(np.zeros((n_batch, n_history, len(task.lexicon))))
            l_msg_next.append(np.zeros((n_batch, n_history, len(task.lexicon))))

        for i in range(n_batch):
            ep = episodes[i]
            for i_agent in range(task.n_agents):
                if ep[0].m1 is not None:
                    h[i_agent][i, :] = ep[0].m1[0][i_agent]
                    z[i_agent][i, :] = ep[0].m1[1][i_agent]
                    l_h[i_agent][i, :] = ep[0].m1[2][i_agent]
                    h_next[i_agent][i, :] = ep[0].m2[0][i_agent]
                    z_next[i_agent][i, :] = ep[0].m2[1][i_agent]
                    l_h_next[i_agent][i, :] = ep[0].m2[2][i_agent]
            for j in range(len(ep)):
                reward[i, j] = ep[j].r
                terminal[i, j] = ep[j].t
                mask[i, j] = 1
                for i_agent in range(task.n_agents):
                    x[i_agent][i, j, :] = ep[j].s1.obs()[i_agent]
                    x_next[i_agent][i, j, :] = ep[j].s2.obs()[i_agent]
                    action[i_agent][i, j, ep[j].a[i_agent]] = 1
                    action_index[i_agent][i, j] = ep[j].a[i_agent]
                    l1 = ep[j].s1.l_msg
                    l2 = ep[j].s2.l_msg
                    l_msg[i_agent][i, j, :] = l1[i_agent]
                    l_msg_next[i_agent][i, j, :] = l2[i_agent]
        assert task.n_agents == 2
        l_msg_target[0] = l_msg_next[1]
        l_msg_target[1] = l_msg_next[0]

        return {
            self.t_reward: reward,
            self.t_terminal: terminal,
            self.t_mask: mask,
            self.t_x: x,
            self.t_x_next: x_next,
            self.t_h: h,
            self.t_h_next: h_next,
            self.t_z: z,
            self.t_z_next: z_next,
            self.t_l_h: l_h,
            self.t_l_h_next: l_h_next,
            self.t_action: action,
            self.t_action_index: action_index,
            self.t_l_msg: l_msg,
            self.t_l_msg_next: l_msg_next,
            self.t_l_msg_target: l_msg_target
        }
