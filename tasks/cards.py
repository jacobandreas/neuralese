import csv
import numpy as np
import os

SIGHT_DIST = 3

class CardsState(object):
    def __init__(self, maze, maze_feats, goal_pos, agent_a_pos, agent_b_pos):
        self.maze = maze
        self.maze_feats = maze_feats
        self.goal_pos = goal_pos
        self.agent_a_pos = agent_a_pos
        self.agent_b_pos = agent_b_pos

    def _obs(self, agent_pos):
        view_feats = self.maze_feats[
                :,
                agent_pos[0]:agent_pos[0]+2*SIGHT_DIST,
                agent_pos[1]:agent_pos[1]+2*SIGHT_DIST]

        pos_feats = [
                1. * agent_pos[0] / self.maze.shape[0], 
                1. * agent_pos[1] / self.maze.shape[1]]

        return np.concatenate((view_feats.ravel(), pos_feats))

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

        if nr < 0 or nc < 0:
            nr, nc = r, c
        elif nr >= self.maze.shape[0] or nc >= self.maze.shape[1]:
            nr, nc = r, c
        elif self.maze[nr, nc] == 1:
            nr, nc = r, c

        return np.asarray([nr, nc])

    def step(self, action_a, action_b):
        #if action_a == 4:
        #    action_a = np.random.randint(4)
        #if action_b == 4:
        #    action_b = np.random.randint(4)
        npos_a = self._move(self.agent_a_pos, action_a)
        npos_b = self._move(self.agent_b_pos, action_b)
        reward = -0.01
        stop = False
        if (npos_a == self.goal_pos).all() or (npos_b == self.goal_pos).all():
            reward = 1
            stop = True
        return (
                CardsState(
                    self.maze, self.maze_feats, self.goal_pos, npos_a, npos_b), 
                reward, stop)

class CardsTask(object):
    def __init__(self):
        maze_strings = set()
        for group in ["01", "02"]:
            for transcript in os.listdir("data/cards/transcripts/%s" % group):
                with open("data/cards/transcripts/%s/%s" % 
                        (group, transcript)) as tfile:
                    lines = tfile.readlines()
                    elines = [l for l in lines if "CREATE_ENVIRONMENT" in l]
                    if len(elines) == 0:
                        continue
                    else:
                        assert len(elines) == 1
                    eline, = elines
                    _, _, _, maze, _, cards = eline.split(",", 5)
                    maze = ";".join(maze.replace("b", " ").split(";")[:-1])[1:]
                    maze_strings.add(maze)

        mstr = sorted(maze_strings, key=len)[0]
        mrows = mstr.split(";")

        maze = np.zeros((len(mrows), len(mrows[0])), dtype=np.int32)
        for r, row in enumerate(mrows):
            for c, ch in enumerate(row):
                maze[r, c] = 1 if ch == "-" else 0
        self.maze = maze
        #self.n_features = maze.shape[0] * maze.shape[1] * 2
        self.n_features = (SIGHT_DIST * 2) ** 2 * 2 + 2
        self.n_actions = 4


    def get_instance(self):
        def get_free(maze, excluded):
            while True:
                r = np.random.randint(maze.shape[0])
                c = np.random.randint(maze.shape[1])
                if maze[r, c] == 1:
                    continue
                if (r, c) in excluded:
                    continue
                return (r, c)

        maze = self.maze
        goal = get_free(maze, [])
        maze_feats = self._make_maze_feats(goal)
        agent_a = get_free(maze, [goal])
        agent_b = get_free(maze, [goal, agent_a])
        return CardsState(
                maze, maze_feats, np.asarray(goal), np.asarray(agent_a),
                np.asarray(agent_b))

    def _make_maze_feats(self, goal_pos):
        pad_maze = np.zeros((
                2,
                self.maze.shape[0] + 2 * SIGHT_DIST,
                self.maze.shape[1] + 2 * SIGHT_DIST))
        pad_maze[
                0,
                SIGHT_DIST:SIGHT_DIST+self.maze.shape[0],
                SIGHT_DIST:SIGHT_DIST+self.maze.shape[1]] = self.maze
        pad_maze[1, SIGHT_DIST+goal_pos[0], SIGHT_DIST+goal_pos[1]] = 1
        return pad_maze
