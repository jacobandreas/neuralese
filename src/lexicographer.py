from experience import Experience
import trainer
from util import Break

import json
import numpy as np
import tensorflow as tf

def kl(d1, d2, w1, w2):
    tot = 0
    for p1, p2, pw1, pw2 in zip(d1.ravel(), d2.ravel(), w1.ravel(), w2.ravel()):
        tot += pw1 * pw2 * p1 * (np.log(p1) - np.log(p2))
    return tot

def skl(d1, d2, w1, w2):
    return kl(d1, d2, w1, w2) + kl(d2, d1, w2, w1)

class Lexicographer(object):
    def __init__(self, states, codes, l_msgs, ph, translator, task, session,
            config):
        self.states = states
        self.distractors = []
        self.codes = codes
        self.l_msgs = l_msgs
        self.translator = translator
        self.task = task
        self.config = config
        self.session = session
        self.ph = ph

        xb = np.zeros((config.trainer.n_batch_episodes, task.n_features))
        xa_true = np.zeros((config.trainer.n_batch_episodes, task.n_features))
        xa_noise = np.zeros(
                (config.trainer.n_batch_episodes, config.trainer.n_distractors,
                    task.n_features))
        for i, state in enumerate(states):
            xb[i, :] = state.obs()[1]
            xa_true[i, :] = state.obs()[0]
            distractors = task.distractors_for(state, 1, config.trainer.n_distractors)
            self.distractors.append(distractors)
            for i_dis in range(len(distractors)):
                dis, _ = distractors[i_dis]
                xa_noise[i, i_dis, :] = dis.obs()[0]

        self.xb = xb
        self.xa_true = xa_true
        self.xa_noise = xa_noise

        all_l_beliefs = []
        all_l_weights = []
        for l_msg in l_msgs:
            l_beliefs, l_weights = self.compute_l_belief(l_msg)
            all_l_beliefs.append(l_beliefs)
            all_l_weights.append(np.tile(
                    l_weights[:, np.newaxis],
                    (1, config.trainer.n_distractors + 1)))
        self.l_beliefs = all_l_beliefs
        self.l_weights = all_l_weights

        all_model_beliefs = []
        all_model_weights = []
        for code in codes:
            model_beliefs, model_weights = self.compute_code_belief(code)
            all_model_beliefs.append(model_beliefs)
            all_model_weights.append(np.tile(
                model_weights[:, np.newaxis],
                (1, config.trainer.n_distractors + 1)))
        self.model_beliefs = all_model_beliefs
        self.model_weights = all_model_weights

    def compute_l_belief(self, l_msg, raw_features=None):
        l_data = np.zeros(
                (self.config.trainer.n_batch_episodes, len(self.task.lexicon)))
        if raw_features is None:
            l_data[:, self.task.lexicon.index(l_msg)] = 1
        else:
            l_data[:, :] = raw_features
        feed = {
            self.ph.t_xb: self.xb,
            self.ph.t_xa_true: self.xa_true,
            self.ph.t_xa_noise: self.xa_noise,
            self.ph.t_l_msg: l_data
        }
        l_beliefs, l_weights = self.session.run(
                [self.translator.t_desc_belief, self.translator.t_desc_weights],
                feed)
        return l_beliefs, l_weights

    def compute_code_belief(self, code):
        code_data = [code] * self.config.trainer.n_batch_episodes
        feed = {
            self.ph.t_xb: self.xb,
            self.ph.t_xa_true: self.xa_true,
            self.ph.t_xa_noise: self.xa_noise,
            self.ph.t_z: code_data
        }
        model_beliefs, model_weights = self.session.run(
                [self.translator.t_model_belief, self.translator.t_model_weights],
                feed)
        return model_beliefs, model_weights

    def l_to_c(self, l_msg):
        if not l_msg.any():
            return np.zeros(self.config.channel.n_msg)
        l_belief, l_weights = self.compute_l_belief(None, raw_features=l_msg)
        if self.config.lexicographer.mode == "belief":
            comparator = skl
        else:
            assert False
        candidates = sorted(
                range(len(self.codes)),
                key=lambda i: comparator(l_belief, self.model_beliefs[i],
                    l_weights, self.model_weights[i]))
        return [self.codes[c] for c in candidates[:10]]

    def c_to_l(self, code):
        if not code.any():
            return [[0]]
        code_belief, code_weights = self.compute_code_belief(code)
        if self.config.lexicographer.mode == "belief":
            comparator = skl
        else:
            assert False
        candidates = sorted(
                range(len(self.l_msgs)),
                key=lambda i: comparator(self.l_beliefs[i], code_belief,
                    self.l_weights[i], code_weights))
        return [self.l_msgs[c] for c in candidates[:10]]

def run(task, rollout_ph, reconst_ph, model, desc_model, translator, session,
        config):
    assert task.n_agents == 2

    h0, z0, _ = session.run(model.zero_state(1, tf.float32))
    states = []
    codes = []
    l_msgs = task.lexicon
    try:
        while True:
            replay = []
            trainer._do_rollout(
                    task, rollout_ph, model, desc_model, replay, [], session,
                    config, 10000, h0, z0, "val")

            for episode in replay:
                for experience in episode[1:]:
                    codes.append(experience.m1[1][config.lexicographer.c_agent])
                    states.append(experience.s1)
                    if len(states) >= config.trainer.n_batch_episodes:
                        raise Break()
    except Break:
        pass

    codes = codes[:50]

    return Lexicographer(
            states, codes, l_msgs, reconst_ph, translator, task, session,
            config)
