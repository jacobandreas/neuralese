#!/usr/bin/env python2

import tasks.cards
import tasks.lock
import tasks.echo
from tasks.data import Transition
import models.dqn

from collections import namedtuple
import logging
import tensorflow as tf
import numpy as np

N_BATCH = 256
N_UPDATE = 100
N_HISTORY = 10
N_HIDDEN = 256
N_CODE = 32

NAME = "indep_cards"
COMMUNICATE = False

task = tasks.cards.CardsTask()
#task = tasks.lock.LockTask()
#task = tasks.echo.EchoTask()
model = models.dqn.DqnModel(
        task, N_BATCH, N_HISTORY, N_HIDDEN, N_CODE, communicate=COMMUNICATE)
session = tf.Session()
session.run(tf.global_variables_initializer())

replay = []
good_replay = []
i_iter = 0

saver = tf.train.Saver()
logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s %(levelname)s\t%(message)s",
        filename="logs/train_%s.log" % NAME, filemode="w")

# rollout
def do_rollout():
    state = task.get_instance()
    rnn_hidden_a = np.zeros((1, N_HIDDEN))
    rnn_hidden_b = np.zeros((1, N_HIDDEN))
    rnn_comm_a = np.zeros((1, N_CODE))
    rnn_comm_b = np.zeros((1, N_CODE))
    transitions = []
    for i in range(100):
        action_a, action_b, rnn_hidden_a_, rnn_hidden_b_, rnn_comm_a_, rnn_comm_b_ = (
                session.run(
                    [model.t_act_a, model.t_act_b, 
                        model.t_act_next_hidden_a, model.t_act_next_hidden_b,
                        model.t_act_next_comm_a, model.t_act_next_comm_b
                    ],
                    {
                        model.t_act_state_a: [state.obs_a()],
                        model.t_act_state_b: [state.obs_b()],
                        model.t_act_hidden_a: rnn_hidden_a,
                        model.t_act_hidden_b: rnn_hidden_b,
                        model.t_act_comm_a: rnn_comm_a,
                        model.t_act_comm_b: rnn_comm_b
                    }))
        mstate = (rnn_hidden_a, rnn_hidden_b, rnn_comm_a, rnn_comm_b)
        mstate_ = (rnn_hidden_a_, rnn_hidden_b_, rnn_comm_a_, rnn_comm_b_)

        rnn_hidden_a = rnn_hidden_a_
        rnn_hidden_b = rnn_hidden_b_
        rnn_comm_a = rnn_comm_a_
        rnn_comm_b = rnn_comm_b_

        action_a = action_a[0]
        action_b = action_b[0]

        eps = max((1000. - i_iter) / 1000., 0.1)
        if np.random.random() < eps:
            action_a = np.random.randint(task.n_actions)
        if np.random.random() < eps:
            action_b = np.random.randint(task.n_actions)

        state_, reward, stop = state.step(action_a, action_b)
        transitions.append(Transition(state, mstate, (action_a, action_b),
                state_, mstate_, reward, stop))
        state = state_
        mstate = mstate_
        if stop:
            break
    replay.append(transitions)
    if sum(max(t.r, 0) for t in transitions) > 0:
        good_replay.append(transitions)

    del replay[:-500]
    del good_replay[:-500]
    return sum(t.r for t in transitions), sum(max(t.r, 0) for t in transitions)

def train_step():
    # update
    experiences = replay + good_replay
    #if len(experiences) < N_BATCH:
    if len(experiences) < 100:
        return 0

    state1_a = np.zeros((N_BATCH, N_HISTORY, task.n_features))
    state1_b = np.zeros((N_BATCH, N_HISTORY, task.n_features))
    state2_a = np.zeros((N_BATCH, N_HISTORY, task.n_features))
    state2_b = np.zeros((N_BATCH, N_HISTORY, task.n_features))
    hidden1_a = np.zeros((N_BATCH, N_HIDDEN))
    hidden1_b = np.zeros((N_BATCH, N_HIDDEN))
    comm1_a = np.zeros((N_BATCH, N_CODE))
    comm1_b = np.zeros((N_BATCH, N_CODE))
    hidden2_a = np.zeros((N_BATCH, N_HIDDEN))
    hidden2_b = np.zeros((N_BATCH, N_HIDDEN))
    comm2_a = np.zeros((N_BATCH, N_CODE))
    comm2_b = np.zeros((N_BATCH, N_CODE))
    action_a = np.zeros((N_BATCH, N_HISTORY, task.n_actions))
    action_b = np.zeros((N_BATCH, N_HISTORY, task.n_actions))
    reward = np.zeros((N_BATCH, N_HISTORY))
    terminal = np.zeros((N_BATCH, N_HISTORY))
    mask = np.zeros((N_BATCH, N_HISTORY))

    for i in range(N_BATCH):
        i_episode = np.random.randint(len(experiences))
        ep = experiences[i_episode]
        i_offset = np.random.randint(len(ep))
        tr = ep[i_offset:i_offset+N_HISTORY]
        assert sum(tr_.r for tr_ in tr) <= 1
        hidden1_a[i, :], hidden1_b[i, :], comm1_a[i, :], comm1_b[i, :] = tr[0].m1
        hidden2_a[i, :], hidden2_b[i, :], comm2_a[i, :], comm2_b[i, :] = tr[0].m2
        for j in range(len(tr)):
            state1_a[i, j, :] = tr[j].s1.obs_a()
            state1_b[i, j, :] = tr[j].s1.obs_b()
            state2_a[i, j, :] = tr[j].s2.obs_a()
            state2_b[i, j, :] = tr[j].s2.obs_b()
            action_a[i, j, tr[j].a[0]] = 1
            action_b[i, j, tr[j].a[1]] = 1
            reward[i, j] = tr[j].r
            terminal[i, j] = tr[j].term
            mask[i, j] = 1

    m_loss, _ = session.run(
        [model.t_loss, model.t_train_op],
        {
            model.t_state1_a: state1_a,
            model.t_state1_b: state1_b,
            model.t_state2_a: state2_a,
            model.t_state2_b: state2_b,
            model.t_init_mstate1: (hidden1_a, hidden1_b, comm1_a, comm1_b),
            model.t_init_mstate2: (hidden2_a, hidden2_b, comm2_a, comm2_b),
            model.t_action_a: action_a,
            model.t_action_b: action_b,
            model.t_reward: reward,
            model.t_terminal: terminal,
            model.t_mask: mask
        })

    return m_loss

total_score = 0.
total_loss = 0.
while i_iter < 100000:
    score = do_rollout()
    loss = train_step()
    total_score += np.asarray(score)
    total_loss += np.asarray(loss)
    i_iter += 1
    if i_iter % N_UPDATE == 0:
        logging.info("[iter]\t%s", i_iter)
        logging.info("[score]\t%s", total_score / N_UPDATE)
        logging.info("[loss]\t%s", total_loss / N_UPDATE)
        logging.info("[eps]\t%s", max((1000. - i_iter) / 1000., 0.1))
        logging.info("")
        total_score = 0.
        total_loss = 0.
        session.run(model.t_update_target_ops)
        saver.save(session, "saves/%s" % NAME)
