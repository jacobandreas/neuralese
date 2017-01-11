from experience import Experience

import logging
import numpy as np
import os
import tensorflow as tf

random = np.random.RandomState(0)

#@profile
def run(task, rollout_ph, replay_ph, reconst_ph, model, translator, session, config):
    saver = tf.train.Saver()
    replay = []
    good_replay = []

    session.run(tf.global_variables_initializer())

    total_score = 0.
    total_loss = 0.
    h0, z0, _ = session.run(
            model.zero_state(config.trainer.n_rollout_episodes, tf.float32))
    for i_iter in range(config.trainer.n_iters):
        score = _do_rollout(
                task, rollout_ph, model, replay, good_replay, session, config,
                i_iter, h0, z0)
        loss = _do_step(
                task, replay_ph, reconst_ph, model, translator, replay,
                good_replay, session, config)
        total_score += np.asarray(score)
        total_loss += np.asarray(loss)

        if (i_iter + 1) % config.trainer.n_update_iters == 0:
            total_score /= config.trainer.n_update_iters
            total_loss /= config.trainer.n_update_iters
            logging.info("[iter] " + "\t%d", i_iter)
            logging.info("[score]" + "\t%2.4f" * total_score.size, *total_score)
            logging.info("[loss] " + "\t%2.4f" * total_loss.size, *total_loss)
            logging.info("")
            total_score = 0.
            total_loss = 0.
            session.run(model.oo_update_target)
            saver.save(session, config.experiment_dir + "/model")

def load(session, config):
    saver = tf.train.Saver()
    saver.restore(session, "experiments/%s/model" % config.model.load)

#@profile
def _do_rollout(
        task, rollout_ph, model, replay, good_replay, session, config, i_iter,
        h0, z0):
    #world = task.get_instance()
    worlds = [task.get_instance() for _ in range(config.trainer.n_rollout_episodes)]
    done = [False] * config.trainer.n_rollout_episodes
    episodes = [[] for i in range(config.trainer.n_rollout_episodes)]
    hs, zs = h0, z0
    for t in range(config.trainer.n_timeout):
        hs_, zs_, qs = session.run(
                [model.tt_rollout_h, model.tt_rollout_z, model.tt_rollout_q],
                rollout_ph.feed(hs, zs, worlds, config))
        eps = max((1000. - i_iter) / 1000., 0.1)
        for i in range(config.trainer.n_rollout_episodes):
            if done[i]:
                continue
            actions = []
            for q in qs:
                q = q[i, :]
                # TODO configurable
                if random.rand() < eps:
                    a = random.randint(len(q))
                else:
                    a = np.argmax(q)
                actions.append(a)

            h = [oh[i] for oh in hs]
            z = [oz[i] for oz in zs]
            h_ = [oh_[i] for oh_ in hs_]
            z_ = [oz_[i] for oz_ in zs_]
            world_, reward, done_ = worlds[i].step(actions)
            episodes[i].append(Experience(
                worlds[i], (h, z), tuple(actions), world_, (h_, z_),
                reward, done_))
            worlds[i] = world_
            done[i] = done_

        hs = hs_
        zs = zs_

        if all(done):
            break

    replay += episodes
    for episode in episodes:
        if any(e.r > 0 for e in episode):
            good_replay.append(episode)

    del replay[:-config.trainer.n_replay_episodes]
    del good_replay[:-config.trainer.n_replay_episodes]
    return (sum(e.r for ep in episodes for e in ep) * 1. / 
                config.trainer.n_rollout_episodes, 
            sum(e.r for ep in episodes for e in ep if e.r > 0) * 1. /
                config.trainer.n_rollout_episodes)

#@profile
def _do_step(
        task, replay_ph, reconst_ph, model, translator, replay, good_replay,
        session, config):
    n_good = int(config.trainer.n_batch_episodes * config.trainer.good_fraction)
    n_any = config.trainer.n_batch_episodes - n_good
    if len(replay) < n_any or len(good_replay) < n_good:
        return [0, 0]
    episodes = []
    for _ in range(n_good):
        episodes.append(good_replay[random.randint(len(good_replay))])
    for _ in range(n_any):
        episodes.append(replay[random.randint(len(replay))])

    slices = []
    for ep in episodes:
        offset = random.randint(len(ep))
        sl = ep[offset:offset+config.trainer.n_batch_history]
        slices.append(sl)

    model_loss, _ = session.run(
            [model.t_loss, model.t_train_op], 
            replay_ph.feed(slices, task, config))

    tr_loss, _ = session.run(
            [translator.t_loss, translator.t_train_op],
            reconst_ph.feed([e[0] for e in slices], 1, 0, task, config))

    desc_belief = session.run(
            [translator.t_desc_belief],
            reconst_ph.feed([e[0] for e in slices], 1, 0, task, config))

    return [model_loss, tr_loss]
