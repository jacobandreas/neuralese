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
    for i_iter in range(config.trainer.n_iters):
        score = _do_rollout(
                task, rollout_ph, model, desc_model, replay, good_replay,
                session, config, i_iter, h0, z0)
        demonstrations.append(task.get_demonstration("train"))
        loss = _do_step(
                task, replay_ph, reconst_ph, model, desc_model, translator,
                replay, good_replay, demonstrations, session, config)
        total_score += np.asarray(score)
        total_loss += np.asarray(loss)

        if (i_iter + 1) % (config.trainer.n_update_iters) == 0:
            total_score /= config.trainer.n_update_iters
            total_loss /= config.trainer.n_update_iters
            logging.info("[iter] " + "\t%d", i_iter)
            logging.info("[score]" + "\t%2.4f" * total_score.size, *total_score)
            logging.info("[loss] " + "\t%2.4f" * total_loss.size, *total_loss)
            logging.info("")
            total_score = 0.
            total_loss = 0.
            session.run(model.oo_update_target)
            #session.run(desc_model.oo_update_target)
            saver.save(session, config.experiment_dir + "/model")

def load(session, config):
    saver = tf.train.Saver()
    saver.restore(session, "experiments/%s/model" % config.model.load)

#@profile
def _do_rollout(
        task, rollout_ph, model, desc_model, replay, good_replay, session,
        config, i_iter, h0, z0, fold="train"):
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
                0.1 * (10000. - i_iter) / 10000.,
                0.)
        for i in range(config.trainer.n_rollout_episodes):
            if done[i]:
                continue
            actions = []
            used_qs = qs
            #used_qs = dqs
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
            sum(e.r for ep in episodes for e in ep if e.r > 0) * 1. /
                config.trainer.n_rollout_episodes)

#@profile
def _do_step(
        task, replay_ph, reconst_ph, model, desc_model, translator, replay,
        good_replay, demonstrations, session, config):
    n_good = int(config.trainer.n_batch_episodes * config.trainer.good_fraction)
    n_any = config.trainer.n_batch_episodes - n_good
    if (len(replay) < n_any or len(good_replay) < n_good
            or len(demonstrations) < config.trainer.n_batch_episodes):
        return [0, 0, 0]
    episodes = []
    desc_episodes = []
    for _ in range(n_good):
        episodes.append(good_replay[random.randint(len(good_replay))])
    for _ in range(n_any):
        episodes.append(replay[random.randint(len(replay))])
    for _ in range(config.trainer.n_batch_episodes):
        desc_episodes.append(demonstrations[random.randint(len(demonstrations))])

    slices = []
    for ep in episodes:
        bd = max(1, len(ep)-config.trainer.n_batch_history)
        offset = random.randint(bd)
        sl = ep[offset:offset+config.trainer.n_batch_history]
        slices.append(sl)

    feed = replay_ph.feed(slices, task, config)
    model_loss, _ = session.run([model.t_loss, model.t_train_op], feed)

    desc_feed = replay_ph.feed(desc_episodes, task, config)
    desc_loss, _ = session.run(
            [desc_model.t_loss, desc_model.t_train_op], desc_feed)

    #if np.random.randint(20) == 0:
    #    print
    #    #print slices[0][0].s1.desc[1]
    #    #print slices[0][0].s2.desc[1]
    #    #print feed[replay_ph.t_z][0][0, :]
    #    #print feed[replay_ph.t_z_next][0][0, :]
    #    print "desc", feed[replay_ph.t_desc][1][0, 0, :]
    #    print "desc n", feed[replay_ph.t_desc_next][1][0, 0, :]
    #    desc_q, desc_q_next, desc_td = session.run([desc_model.t_q[1], 
    #        desc_model.t_q_next[1], desc_model.t_td[1]], feed)
    #    print "q", desc_q[0, ...] * feed[replay_ph.t_mask][0, :, np.newaxis]
    #    print "q n", desc_q_next[0, ...] * feed[replay_ph.t_mask][0, :, np.newaxis]
    #    print "td", desc_td[0, ...] * feed[replay_ph.t_mask][0, :]
    #    print "answer", slices[0][0].s1.target, slices[0][0].s2.target
    #    #print desc_td[0, ...] * feed[replay_ph.t_mask][0, :]
    #    db1, db2, db3 = session.run([desc_model.t_debug_1, desc_model.t_debug_2,
    #        desc_model.t_debug_3], feed)
    #    print "fut", db1[0, 0]
    #    print "rew", db2[0, 0]
    #    print "chosen", db3[0, 0]
    #    #print np.mean(feed[replay_ph.t_mask], axis=0)

    tr_loss, _ = session.run(
            [translator.t_loss, translator.t_train_op],
            reconst_ph.feed([e[random.randint(len(e))] for e in slices], 1, 0, task, config))

    return [model_loss, desc_loss, tr_loss]
