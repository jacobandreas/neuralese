import numpy as np

class RefState(object):
    def __init__(self, left, right, target):
        self.left = left
        self.right = right
        self.target = target
        assert target in (0, 1), "invalid target ID"

    # speaker observation
    def obs(self):
        return (self.obs_a(), self.obs_b())

    def obs_a(self):
        if self.target == 0:
            return np.concatenate((self.left, self.right))
        else:
            return np.concatenate((self.right, self.left))

    # listener observation
    def obs_b(self):
        return np.concatenate((self.left, self.right))

    def step(self, actions):
        assert len(actions) == 2
        action_a, action_b = actions
        assert action_a == 0
        if action_b < 2:
            reward = 1 if action_b == self.target else 0
            return self, reward, True
        return self, -.1, False

class RefTask(object):
    def __init__(self):
        self.n_agents = 2
        self.n_actions = (1, 3)
        self.symmetric = False
        self.random = np.random.RandomState(0)

    def get_instance(self):
        target, distractor = self.get_pair()
        if self.random.rand() < 0.5:
            return RefState(target, distractor, 0)
        else:
            return RefState(distractor, target, 1)
