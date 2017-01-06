from collections import namedtuple
import numpy as np
import tensorflow as tf

Experience = namedtuple("Experience", ["s1", "m1", "a", "s2", "m2", "r"])

class RolloutPlaceholders(object):
    def __init__(self, task, config):
        t_x = []
        t_h = []
        t_z = []
        t_q = []
        for i_agent in range(task.n_agents):
            t_x.append(tf.placeholder(tf.float32, (1, task.n_features)))
            t_h.append(tf.placeholder(tf.float32, (1, config.model.n_hidden)))
            t_z.append(tf.placeholder(tf.float32, (1, config.channel.n_msg)))
            t_q.append(tf.zeros((1, task.n_actions[i_agent]), tf.float32))
        self.t_x = tuple(t_x)
        self.t_h = tuple(t_h)
        self.t_z = tuple(t_z)
        self.t_q = tuple(t_q)

    def feed(self, h, z, world):
        out = {}
        # TODO why is this necessary?
        out.update({t: [x] for t, x in zip(self.t_x, world.obs())})
        out[self.t_h] = h
        out[self.t_z] = z
        return out

class ReplayPlaceholders(object):
    def __init__(self, task, config):
        self.t_reward = tf.placeholder(tf.float32,
                (config.trainer.n_batch_episodes,
                    config.trainer.n_batch_history))
        self.t_terminal = tf.placeholder(tf.float32,
                (config.trainer.n_batch_episodes,
                    config.trainer.n_batch_history))
        self.t_mask = tf.placeholder(tf.float32,
                (config.trainer.n_batch_episodes,
                    config.trainer.n_batch_history))

        t_x = []
        t_x_next = []
        t_z = []
        t_z_next = []
        t_h = []
        t_h_next = []
        t_q = []
        t_q_next = []
        t_action = []
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

            t_q.append(tf.zeros(
                    (config.trainer.n_batch_episodes, task.n_actions[i_agent]),
                    tf.float32))
            t_q_next.append(tf.zeros(
                    (config.trainer.n_batch_episodes, task.n_actions[i_agent]),
                    tf.float32))

            t_action.append(tf.placeholder(
                    tf.float32,
                    (config.trainer.n_batch_episodes,
                        config.trainer.n_batch_history,
                        task.n_actions[i_agent])))
        self.t_x = tuple(t_x)
        self.t_x_next = tuple(t_x_next)
        self.t_z = tuple(t_z)
        self.t_z_next = tuple(t_z_next)
        self.t_h = tuple(t_h)
        self.t_h_next = tuple(t_h_next)
        self.t_q = tuple(t_q)
        self.t_q_next = tuple(t_q_next)
        self.t_action = tuple(t_action)

    def feed(self, episodes):
        out = {}
