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
        loss = _do_step(
                task, replay_ph, reconst_ph, model, desc_model, translator,
                replay, good_replay, session, config)
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
            session.run(desc_model.oo_update_target)
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
            #print worlds[i].obs()[0]
            #print worlds[i].obs()[1]
            #print dqs
            #exit()
            actions = []
            if random.randint(2) == 0:
                used_qs = qs
            else:
                used_qs = dqs
            #used_qs = dqs
            for q in used_qs:
                q = q[i, :]
                # TODO configurable
                if random.rand() < eps:
                    a = random.randint(len(q))
                else:
                    a = np.argmax(q)
                actions.append(a)

            #print actions

            h = [oh[i] for oh in hs]
            z = [oz[i] for oz in zs]
            dh = [odh[i] for odh in dhs]
            h_ = [oh_[i] for oh_ in hs_]
            z_ = [oz_[i] for oz_ in zs_]
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
        good_replay, session, config):
    n_good = int(config.trainer.n_batch_episodes * config.trainer.good_fraction)
    n_any = config.trainer.n_batch_episodes - n_good
    if len(replay) < n_any or len(good_replay) < n_good:
        return [0, 0, 0]
    episodes = []
    for _ in range(n_good):
        episodes.append(good_replay[random.randint(len(good_replay))])
    for _ in range(n_any):
        episodes.append(replay[random.randint(len(replay))])

    slices = []
    for ep in episodes:
        bd = max(1, len(ep)-config.trainer.n_batch_history)
        offset = random.randint(bd)
        sl = ep[offset:offset+config.trainer.n_batch_history]
        slices.append(sl)

    feed = replay_ph.feed(slices, task, config)
    model_loss, _ = session.run([model.t_loss, model.t_train_op], feed)
    desc_loss, _ = session.run([desc_model.t_loss, desc_model.t_train_op], feed)

    ### #if np.random.randint(20) == 0:
    ### if False:
    ###     #print slices[0][0].s1.desc
    ###     #print slices[0][0].s2.desc
    ###     #print feed[replay_ph.t_z][0][0, :]
    ###     #print feed[replay_ph.t_z_next][0][0, :]
    ###     print
    ###     #print feed[replay_ph.t_desc][1][0, 0, :]
    ###     #print feed[replay_ph.t_desc_next][1][0, 0, :]
    ###     desc_q, desc_q_next, desc_td = session.run([desc_model.t_q[1], 
    ###         desc_model.t_q_next[1], desc_model.t_td[1]], feed)
    ###     print desc_q[0, ...] * feed[replay_ph.t_mask][0, :, np.newaxis]
    ###     print desc_q_next[0, ...] * feed[replay_ph.t_mask][0, :, np.newaxis]
    ###     #print desc_td[0, ...] * feed[replay_ph.t_mask][0, :]
    ###     #db1, db2, db3 = session.run([desc_model.t_debug_1, desc_model.t_debug_2,
    ###     #    desc_model.t_debug_3], feed)
    ###     #print db1[0, 0]
    ###     #print db2[0, 0]
    ###     #print db3[0, 0]
    ###     #print np.mean(feed[replay_ph.t_mask], axis=0)

    tr_loss, _ = session.run(
            [translator.t_loss, translator.t_train_op],
            reconst_ph.feed([e[random.randint(len(e))] for e in slices], 1, 0, task, config))

    return [model_loss, desc_loss, tr_loss]
