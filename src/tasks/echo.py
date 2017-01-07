import numpy as np

N_SYMBOLS = 2

class EchoState(object):
    def __init__(self, a_sym, b_sym):
        self.a_sym = a_sym
        self.b_sym = b_sym

    def obs(self):
        return (self.obs_a(), self.obs_b())

    def obs_a(self):
        v = np.zeros(N_SYMBOLS)
        v[self.a_sym] = 1
        return v

    def obs_b(self):
        v = np.zeros(N_SYMBOLS)
        v[self.b_sym] = 1
        return v

    def step(self, actions):
        action_a, action_b = actions
        if action_a == N_SYMBOLS or action_b == N_SYMBOLS:
            return self, -0.01, False
        if action_a == self.b_sym and action_b == self.a_sym:
            return self, 1, True
        return self, 0, True

class EchoTask(object):
    def __init__(self):
        self.n_features = N_SYMBOLS
        self.n_actions = (N_SYMBOLS + 1,) * 2
        self.n_agents = 2
        self.symmetric = True

    def get_instance(self):
        a_sym = np.random.randint(N_SYMBOLS)
        b_sym = np.random.randint(N_SYMBOLS)
        return EchoState(a_sym, b_sym)
