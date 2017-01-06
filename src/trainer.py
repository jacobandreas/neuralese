from experience import Experience

import logging
import numpy as np
import tensorflow as tf

random = np.random.RandomState(0)

def run(task, rollout_ph, replay_ph, model, session, config):
    saver = tf.train.Saver()
    replay = []
    good_replay = []

    session.run(tf.global_variables_initializer())

    total_score = 0.
    total_loss = 0.
    for i_iter in range(config.trainer.n_iters):
        score = _do_rollout(
                task, rollout_ph, model, replay, good_replay, session, config)
        loss = _do_step(
                task, replay_ph, model, replay, good_replay, session, config)
        total_score += np.asarray(score)
        total_loss += np.asarray(loss)

        if (i_iter + 1) % config.trainer.n_update_iters == 0:
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
    saver.load(session, config.experiment_dir + "/model")

def _do_rollout(task, rollout_ph, model, replay, good_replay, session, config):
    hs, zs, _ = session.run(model.zero_state(1, tf.float32))
    world = task.get_instance()
    episode = []
    for t in range(config.trainer.n_timeout):
        hs_, zs_, qs = session.run(
                [model.tt_rollout_h, model.tt_rollout_z, model.tt_rollout_q],
                rollout_ph.feed(hs, zs, world))
        actions = []
        for (q,) in qs:
            if random.rand() < 0.1:
                a = random.randint(len(q))
            else:
                a = np.argmax(q)
            actions.append(a)

        world_, reward, done = world.step(actions)
        episode.append(Experience(
            world.obs(), (hs, zs), tuple(actions), world_.obs(), (hs_, zs_),
            reward))
        world = world_
        hs = hs_
        zs = zs_

        if done:
            break

    replay.append(episode)
    if any(e.r > 0 for e in episode):
        good_replay.append(episode)

    return (sum(e.r for e in episode), sum(e.r for e in episode if e.r > 0))

def _do_step(task, replay_ph, model, replay, good_replay, session, config):
    n_good = config.trainer.n_batch_episodes * config.trainer.good_fraction
    n_any = config.trainer.n_batch_episodes - n_good

    if len(replay) < n_any or len(good_replay) < n_good:
        return [0]

    episodes = []
    for _ in range(n_good):
        episodes.append(good_replay[random.randint(len(good_replay))])
    for _ in range(n_any):
        episodes.append(replay[random.randint(len(replay))])

    loss, _ = session.run(
            [model.t_loss, model.t_train_op], replay_ph.feed(episodes))

    del replay[:-config.trainer.n_replay_episodes]
    del good_replay[:-config.trainer.n_replay_episodes]

    return loss
