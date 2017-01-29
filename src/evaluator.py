from experience import Experience
import trainer

import json
import logging
import numpy as np
import tensorflow as tf


def _do_tr_rollout(
        code_agent, desc_agent, task, rollout_ph, model, desc_model,
        desc_to_code, code_to_desc, session, config, h0, z0, fold, mode):
    worlds = [task.get_instance(fold) for _ in range(config.trainer.n_rollout_episodes)]
    done = [False] * config.trainer.n_rollout_episodes
    episodes = [[] for i in range(config.trainer.n_rollout_episodes)]
    hs, zs = h0, z0
    dhs = h0
    for t in range(config.trainer.n_timeout):
        hs_, zs_, qs = session.run(
                [model.tt_rollout_h, model.tt_rollout_z, model.tt_rollout_q],
                rollout_ph.feed(hs, zs, dhs, worlds, task, config))
        dhs_, dqs = session.run(
                [desc_model.tt_rollout_h, desc_model.tt_rollout_q],
                rollout_ph.feed(hs, zs, dhs, worlds, task, config))
        for i in range(config.trainer.n_rollout_episodes):
            if done[i]:
                continue

            actions = [None, None]
            actions[code_agent] = np.argmax(qs[code_agent][i, :])
            actions[desc_agent] = np.argmax(dqs[desc_agent][i, :])

            world_, reward, done_ = worlds[i].step(actions)

            code = desc_to_code(world_.l_msg[code_agent], mode)[0]
            zs_[desc_agent][i, :] = code

            l_words = code_to_desc(zs_[code_agent][i, :], mode)[:5]
            l_msg = np.zeros(len(task.lexicon))
            for l_word in l_words:
                l_msg[task.lexicon.index(l_word)] += 1
            l_msg /= np.sum(l_msg)

            world_.l_msg = list(world_.l_msg)
            world_.l_msg[desc_agent] = l_msg
            world_.l_msg = tuple(world_.l_msg)

            episodes[i].append(Experience(
                worlds[i], None, tuple(actions), world_, None, reward, done_))
            worlds[i] = world_
            done[i] = done_

            if config.evaluator.simulate_l:
                assert False

        hs = hs_
        zs = zs_
        dhs = dhs_
        if all(done):
            break

    return (sum(e.r for ep in episodes for e in ep) * 1. / 
                config.trainer.n_rollout_episodes, 
            sum(ep[-1].s2.success for ep in episodes) * 1. /
                config.trainer.n_rollout_episodes)

def run(task, rollout_ph, replay_ph, reconst_ph, model, desc_model,
        lexicographer, session, config, fold="test"):
    h0, z0, _ = session.run(model.zero_state(1, tf.float32))

    count = config.evaluator.n_episodes

    with open(config.experiment_dir + "/eval.txt", "w") as eval_f:
        task.reset_test()
        l_l_score = np.asarray([0., 0.])
        for i in range(count):
            score = trainer._do_rollout(
                    task, rollout_ph, model, desc_model, [], [], session,
                    config, 10000, h0, z0, fold, use_desc=True)
            l_l_score += score
        l_l_score /= count
        logging.info("[l,l]  \t%s" % str(l_l_score))
        print >>eval_f, "l only:"
        print >>eval_f, l_l_score

        task.reset_test()
        c_c_score = np.asarray([0., 0.])
        for i in range(count):
            score = trainer._do_rollout(
                    task, rollout_ph, model, desc_model, [], [], session,
                    config, 10000, h0, z0, fold, use_desc=False)
            c_c_score += score
        c_c_score /= count
        logging.info("[c,c]  \t%s\n" % str(c_c_score))
        print >>eval_f, "c only:"
        print >>eval_f, c_c_score

        for mode in ["fkl", "rkl", "pmi", "dot"]:
            task.reset_test()
            c_l_score = np.asarray([0., 0.])
            for i in range(count):
                score = _do_tr_rollout(
                        0, 1, task, rollout_ph, model, desc_model, lexicographer.l_to_c,
                        lexicographer.c_to_l, session, config, h0, z0, fold, mode)
                c_l_score += score
            c_l_score /= count
            logging.info("[c,l:%s]  \t%s" % (mode, str(c_l_score)))

            task.reset_test()
            l_c_score = np.asarray([0., 0.])
            for i in range(count):
                score = _do_tr_rollout(
                        1, 0, task, rollout_ph, model, desc_model, lexicographer.l_to_c,
                        lexicographer.c_to_l, session, config, h0, z0, fold, mode)
                l_c_score += score
            l_c_score /= count
            logging.info("[l,c:%s]  \t%s" % (mode, str(l_c_score)))
            logging.info("")

            print >>eval_f, mode + ":"
            print >>eval_f, "(c, l)", c_l_score
            print >>eval_f, "(l, c)", l_c_score
