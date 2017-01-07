import csv
import numpy as np
import os

SIGHT_DIST = 3

class LockState(object):
    def __init__(self, board, goal_1_pos, goal_2_pos, agent_a_pos, agent_b_pos):
        self.board = board
        self.goal_1_pos = goal_1_pos
        self.goal_2_pos = goal_2_pos
        self.agent_a_pos = agent_a_pos
        self.agent_b_pos = agent_b_pos

    def _obs(self, agent_pos):
        feats = np.zeros((2,) + self.board.shape)
        feats[0, agent_pos[0], agent_pos[1]] = 1
        feats[1, self.goal_1_pos[0], self.goal_1_pos[1]] = 1
        feats[1, self.goal_2_pos[0], self.goal_2_pos[1]] = 1
        return feats.ravel()

    def obs_a(self):
        return self._obs(self.agent_a_pos)

    def obs_b(self):
        return self._obs(self.agent_b_pos)

    def _move(self, pos, action):
        r, c = pos
        if action == 0:
            nr, nc = r - 1, c
        elif action == 1:
            nr, nc = r, c - 1
        elif action == 2:
            nr, nc = r + 1, c
        elif action == 3:
            nr, nc = r, c + 1
        elif action == 4:
            nr, nc = r, c

        if nr < 0 or nc < 0:
            nr, nc = r, c
        elif nr >= self.board.shape[0] or nc >= self.board.shape[1]:
            nr, nc = r, c
        elif self.board[nr, nc] == 1:
            nr, nc = r, c

        return np.asarray([nr, nc])

    def step(self, action_a, action_b):
        npos_a = self._move(self.agent_a_pos, action_a)
        npos_b = self._move(self.agent_b_pos, action_b)

        reward = -0.01
        stop = False

        if (((npos_a == self.goal_1_pos).all() and 
                (npos_b == self.goal_2_pos).all()) or
                ((npos_b == self.goal_1_pos).all() and 
                (npos_a == self.goal_2_pos).all())):
            reward = 1
            stop = True

        return LockState(self.board, self.goal_1_pos, self.goal_2_pos, npos_a,
                npos_b), reward, stop

class LockTask(object):
    def __init__(self):
        self.board = np.zeros((5, 5))
        self.n_features = self.board.shape[0] * self.board.shape[1] * 2
        self.n_actions = 5
        self.random = np.random.RandomState(0)

    def get_instance(self):
        def get_free(board, excluded):
            while True:
                r = self.random.randint(board.shape[0])
                c = self.random.randint(board.shape[1])
                if board[r, c] == 1:
                    continue
                if (r, c) in excluded:
                    continue
                return (r, c)

        board = self.board
        goal_1 = get_free(board, [])
        goal_2 = get_free(board, [goal_1])
        agent_a = get_free(board, [goal_1, goal_2])
        agent_b = get_free(board, [goal_1, goal_2, agent_a])
        return LockState(board, np.asarray(goal_1), np.asarray(goal_2),
            np.asarray(agent_a), np.asarray(agent_b))
