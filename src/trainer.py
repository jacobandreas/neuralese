from experience import Experience

import logging
import numpy as np
import os
import tensorflow as tf

random = np.random.RandomState(0)

#@profile
def run(task, rollout_ph, replay_ph, reconst_ph, model, desc_model, translator,
        session, config):
    saver = tf.train.Saver()
    replay = []
    good_replay = []
    demonstrations = []

    if config.trainer.resume:
        load(session, config)
    else:
        session.run(tf.global_variables_initializer())

    total_score = 0.
    total_loss = 0.
    h0, z0, _ = session.run(
            model.zero_state(config.trainer.n_rollout_episodes, tf.float32))
    max_iters = max(config.trainer.n_iters, config.trainer.n_desc_iters)
    for i_iter in range(max_iters):
        score = _do_rollout(
                task, rollout_ph, model, desc_model, replay, good_replay,
                session, config, i_iter, h0, z0)
        demonstrations.append(task.get_demonstration("train"))
        loss = _do_step(
                task, replay_ph, reconst_ph, model, desc_model, translator,
                replay, good_replay, demonstrations, session, config,
                i_iter < config.trainer.n_iters,
                i_iter < config.trainer.n_desc_iters)
        total_score += np.asarray(score)
        total_loss += np.asarray(loss)

        if (i_iter + 1) % (config.trainer.n_update_iters) == 0:
            total_score /= config.trainer.n_update_iters
            total_loss /= config.trainer.n_update_iters
            logging.info("[iter] " + "\t%d", i_iter+1)
            logging.info("[score]" + "\t%2.4f" * total_score.size, *total_score)
            logging.info("[loss] " + "\t%2.4f" * total_loss.size, *total_loss)
            logging.info("")
            total_score = 0.
            total_loss = 0.
            session.run(model.oo_update_target)

            #if not replay[-1][-1].s2.success:
            #    logging.info("\n" + task.visualize(replay[-1][0].s1, 0))
            #    logging.info("\n" + task.visualize(replay[-1][0].s1, 1))
            #    logging.info("")
            #    logging.info(str([c.goal for c in replay[-1][0].s1.cars]))
            #    logging.info(str([c.pos for c in replay[-1][0].s1.cars]))
            #    for tr in replay[-1]:
            #        logging.info(str([c.pos for c in tr.s2.cars]))

            if (i_iter + 1) % (10 * config.trainer.n_update_iters) == 0:
                saver.save(session, config.experiment_dir + "/model")

            #if (i_iter + 1) % (10 * config.trainer.n_update_iters) == 0:
            #if True:
            if False:
                import lexicographer
                import evaluator
                import calibrator
                lex = lexicographer.run(
                        task, rollout_ph, reconst_ph, model, desc_model,
                        translator, session, config)
                calibrator.run(
                        task, rollout_ph, model, desc_model, lex, session,
                        config)
                evaluator.run(
                        task, rollout_ph, replay_ph, reconst_ph, model,
                        desc_model, lex, session, config, "val")

def load(session, config):
    saver = tf.train.Saver()
    saver.restore(session, "experiments/%s/model" % config.model.load)

#@profile
def _do_rollout(
        task, rollout_ph, model, desc_model, replay, good_replay, session,
        config, i_iter, h0, z0, fold="train", use_desc=False):
    worlds = [task.get_instance(fold) for _ in range(config.trainer.n_rollout_episodes)]
    done = [False] * config.trainer.n_rollout_episodes
    episodes = [[] for i in range(config.trainer.n_rollout_episodes)]
    hs, zs, dhs = h0, z0, h0
    for t in range(config.trainer.n_timeout):
        hs_, zs_, qs, dhs_, dqs = session.run(
                [model.tt_rollout_h, model.tt_rollout_z, model.tt_rollout_q,
                    desc_model.tt_rollout_h, desc_model.tt_rollout_q],
                rollout_ph.feed(hs, zs, dhs, worlds, task, config))
        eps = max(
                (1000. - i_iter) / 1000., 
                #0.1 * (10000. - i_iter) / 10000.,
                0.1 * (5000. - i_iter) / 5000.,
                #0.01)
                0)
        for i in range(config.trainer.n_rollout_episodes):
            if done[i]:
                continue
            actions = []
            used_qs = qs
            if use_desc:
                used_qs = dqs
            for q in used_qs:
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
            dh = [odh[i] for odh in dhs]
            dh_ = [odh_[i] for odh_ in dhs_]
            world_, reward, done_ = worlds[i].step(actions)
            done_ = done_ or t == config.trainer.n_timeout - 1
            episodes[i].append(Experience(
                worlds[i], (h, z, dh), tuple(actions), world_, (h_, z_, dh_),
                reward, done_))
            worlds[i] = world_
            done[i] = done_

        hs = hs_
        zs = zs_
        dhs = dhs_

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
            sum(ep[-1].s2.success for ep in episodes) * 1. /
                config.trainer.n_rollout_episodes)

#@profile
def _do_step(
        task, replay_ph, reconst_ph, model, desc_model, translator, replay,
        good_replay, demonstrations, session, config, update_model, update_desc):
    n_good = int(config.trainer.n_batch_episodes * config.trainer.good_fraction)
    n_any = config.trainer.n_batch_episodes - n_good
    if (len(replay) < n_any or len(good_replay) < n_good
            or len(demonstrations) < config.trainer.n_batch_episodes):
        return [0, 0, 0, 0]
    episodes = []
    desc_episodes = []
    for _ in range(n_good):
        episodes.append(good_replay[random.randint(len(good_replay))])
    for _ in range(n_any):
        episodes.append(replay[random.randint(len(replay))])
    for _ in range(config.trainer.n_batch_episodes):
        demo = demonstrations[random.randint(len(demonstrations))]
        offset = np.random.randint(max(1, len(demo)-config.trainer.n_batch_history))
        sl = demo[offset:offset+config.trainer.n_batch_history]
        desc_episodes.append(sl)

    slices = []
    for ep in episodes:
        bd = max(1, len(ep)-config.trainer.n_batch_history)
        offset = random.randint(bd)
        sl = ep[offset:offset+config.trainer.n_batch_history]
        slices.append(sl)

    model_loss = tr_m_loss = tr_d_loss = 0
    if update_model:
        feed = replay_ph.feed(slices, task, config)
        model_loss, _ = session.run([model.t_loss, model.t_train_op], feed)

        tr_m_feed = reconst_ph.feed(
                [e[random.randint(len(e))] for e in slices], 1, 0, task, config)
        tr_m_loss, _ = session.run(
                [translator.t_model_loss, translator.t_train_model_op], tr_m_feed)
        tr_d_feed = reconst_ph.feed(
                [e[random.randint(len(e))] for e in desc_episodes], 1, 0, task,
                config)
        tr_d_loss, _ = session.run(
                [translator.t_desc_loss, translator.t_train_desc_op], tr_d_feed)

    desc_loss = 0
    if update_desc:
        desc_feed = replay_ph.feed(desc_episodes, task, config)
        desc_loss, _ = session.run(
                [desc_model.t_loss, desc_model.t_train_op], desc_feed)

    return [model_loss, desc_loss, tr_m_loss, tr_d_loss]
