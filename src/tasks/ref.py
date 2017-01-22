from experience import Experience

import numpy as np

class RefState(object):
    def __init__(self, left, right, target, desc, left_data, right_data, first):
        self.left = left
        self.right = right
        self.target = target
        self.left_data = left_data
        self.right_data = right_data
        self.desc = ([], []) if first else ([], desc)
        self.real_desc = desc
        self.first = first
        assert target in (0, 1), "invalid target ID"

    # speaker observation
    def obs(self):
        return (self.obs_a(), self.obs_b())

    def obs_a(self):
        if self.target == 0:
            return np.concatenate((self.left, self.right, [self.first]))
        else:
            return np.concatenate((self.right, self.left, [self.first]))

    # listener observation
    def obs_b(self):
        if self.first:
            return np.concatenate(
                    (np.zeros(self.left.shape), np.zeros(self.right.shape),
                        [self.first]))
        return np.concatenate((self.left, self.right, [self.first]))

    def step(self, actions):
        assert len(actions) == 2
        action_a, action_b = actions
        assert action_a == 0
        succ = RefState(
                self.left, self.right, self.target, self.real_desc,
                self.left_data, self.right_data, first=False)
        if self.first and action_b < 2:
            return succ, 0, True
        if action_b < 2:
            reward = 1 if action_b == self.target else 0
            return succ, reward, True
        return succ, -.1, False

class RefTask(object):
    def __init__(self, n_example_features):
        self.n_agents = 2
        self.n_actions = (1, 3)
        self.symmetric = False
        self.random = np.random.RandomState(0)
        self.n_features = n_example_features * 2 + 1

    def get_instance(self, fold):
        target, distractor, desc, left_data, right_data = self.get_pair(fold)
        if self.random.rand() < 0.5:
            return RefState(
                    target, distractor, 0, desc, left_data, right_data,
                    first=True)
        else:
            return RefState(
                    distractor, target, 1, desc, left_data, right_data,
                    first=True)

    def get_demonstration(self, fold):
        state1 = self.get_instance(fold)
        action1 = (0, 2)
        state2, r1, _ = state1.step(action1)
        action2 = (0, state1.target)
        state3, r2, _ = state2.step(action2)
        assert r2 == 1
        ep = []
        ep.append(Experience(
            state1, None, action1, state2, None, r1, False))
        ep.append(Experience(
            state2, None, action2, state3, None, r2, True))
        return ep

    def distractors_for(self, state, obs_agent, n_samples):
        out = []
        for i in range(n_samples):
            tgt = state.target
            if obs_agent == 1:
                tgt = (i + 1 - state.target) % 2
            out.append((
                RefState(
                    state.left, state.right, tgt, None, state.left_data,
                    state.right_data, state.first),
                0.5))
        return out
