from experience import Experience
import trainer

import json
import numpy as np
import tensorflow as tf

class Break(Exception):
    pass

def kl(d1, d2, w1, w2):
    tot = 0
    for p1, p2, pw1, pw2 in zip(d1.ravel(), d2.ravel(), w1.ravel(), w2.ravel()):
        tot += pw1 * pw2 * p1 * (np.log(p1) - np.log(p2))
    return tot

def skl(d1, d2, w1, w2):
    #return kl(d1, d2, w1, w2) + kl(d2, d1, w2, w1)
    return kl(d2, d1, w2, w1)

def _do_tr_rollout(
        task, rollout_ph, model, desc_model, desc_to_code, code_to_desc,
        session, config, h0, z0, fold="val"):
    code_agent = config.evaluator.code_agent
    desc_agent = config.evaluator.desc_agent
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
            code = desc_to_code(world_.desc[desc_agent])
            desc = code_to_desc(zs_[code_agent][i, :])
            #world_.desc = desc
            zs_[desc_agent][i, :] = code

            episodes[i].append(Experience(
                worlds[i], None, tuple(actions), world_, None, reward, done_))
            worlds[i] = world_
            done[i] = done_

            if not done_:
                print task.pp(desc), task.pp(world_.real_desc)

        hs = hs_
        zs = zs_
        dhs = dhs_
        if all(done):
            break

    return (sum(e.r for ep in episodes for e in ep) * 1. / 
                config.trainer.n_rollout_episodes, 
            sum(e.r for ep in episodes for e in ep if e.r > 0) * 1. /
                config.trainer.n_rollout_episodes)

def run(task, rollout_ph, replay_ph, reconst_ph, model, desc_model, translator,
        session, config):
    assert task.n_agents == 2

    h0, z0, _ = session.run(model.zero_state(1, tf.float32))
    codes = []
    states = []
    descs = []
    try:
        while True:
            replay = []
            trainer._do_rollout(
                    task, rollout_ph, model, desc_model, replay, [], session,
                    config, 10000, h0, z0, "val")

            for episode in replay:
                for experience in episode[1:]:
                    codes.append(experience.m1[1][config.evaluator.code_agent])
                    states.append(experience.s1)
                    descs.append(tuple(experience.s1.desc[config.evaluator.desc_agent]))
                    if len(states) >= config.trainer.n_batch_episodes:
                        raise Break()
    except Break:
        pass

    assert len(codes) == len(states) == len(descs) == config.trainer.n_batch_episodes
    descs = list(set(descs))

    xb = np.zeros((config.trainer.n_batch_episodes, task.n_features))
    xa_true = np.zeros((config.trainer.n_batch_episodes, task.n_features))
    xa_noise = np.zeros(
            (config.trainer.n_batch_episodes, config.trainer.n_distractors,
                task.n_features))
    for i, state in enumerate(states):
        xb[i, :] = state.obs()[1]
        xa_true[i, :] = state.obs()[0]
        distractors = task.distractors_for(state, config.trainer.n_distractors)
        distractor_vis = []
        for i_dis in range(len(distractors)):
            dis, _ = distractors[i_dis]
            xa_noise[i, i_dis, :] = dis.obs()[0]
            distractor_vis.append(task.visualize(dis, 0))

    def compute_desc_belief(desc):
        desc_data = np.zeros(
                (config.trainer.n_batch_episodes, task.max_desc_len))
        desc_data[:, :len(desc)] = desc
        feed = {
            reconst_ph.t_xb: xb,
            reconst_ph.t_xa_true: xa_true,
            reconst_ph.t_xa_noise: xa_noise,
            reconst_ph.t_desc: desc_data
        }
        desc_beliefs, desc_weights = session.run(
                [translator.t_desc_belief, translator.t_desc_weights], feed)
        return desc_beliefs, desc_weights

    def compute_code_belief(code):
        code_data = [code] * config.trainer.n_batch_episodes
        feed = {
            reconst_ph.t_xb: xb,
            reconst_ph.t_xa_true: xa_true,
            reconst_ph.t_xa_noise: xa_noise,
            reconst_ph.t_z: code_data
        }
        model_beliefs, model_weights = session.run(
                [translator.t_model_belief, translator.t_model_weights], feed)
        return model_beliefs, model_weights

    all_desc_beliefs = []
    all_desc_weights = []
    for desc in descs:
        desc_beliefs, desc_weights = compute_desc_belief(desc)
        all_desc_beliefs.append(desc_beliefs)
        all_desc_weights.append(np.tile(
                desc_weights[:, np.newaxis],
                (1, config.trainer.n_distractors + 1)))

    all_model_beliefs = []
    all_model_weights = []
    for code in codes:
        model_beliefs, model_weights = compute_code_belief(code)
        all_model_beliefs.append(model_beliefs)
        all_model_weights.append(np.tile(
            model_weights[:, np.newaxis],
            (1, config.trainer.n_distractors + 1)))

    def desc_to_code(desc):
        if desc == []:
            return np.zeros(config.channel.n_msg)
        desc_belief, desc_weights = compute_desc_belief(desc)
        candidates = sorted(
                range(len(codes)),
                key=lambda i: skl(desc_belief, all_model_beliefs[i],
                    desc_weights, all_model_weights[i]))
        return codes[candidates[0]]

    def code_to_desc(code):
        if not code.any():
            return []
        code_belief, code_weights = compute_code_belief(code)
        candidates = sorted(
                range(len(descs)),
                key=lambda i: skl(all_desc_beliefs[i], code_belief,
                    all_desc_weights[i], code_weights))
        return descs[candidates[0]]

    task.random = np.random.RandomState(0)

    tot = np.asarray([0., 0.])
    for i in range(100):
        score = _do_tr_rollout(
                task, rollout_ph, model, desc_model, desc_to_code, code_to_desc,
                session, config, h0, z0)
        print score
        tot += score
    print tot / 100
