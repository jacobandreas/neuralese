from experience import Experience
import trainer

import json
import numpy as np
import tensorflow as tf

class Break(Exception):
    pass

def run(task, rollout_ph, replay_ph, reconst_ph, model, desc_model, translator,
        session, config):
    h0, z0, _ = session.run(model.zero_state(1, tf.float32))

    i_rollout = 0
    codes = []
    states = []
    descs = []
    while len(codes) < config.trainer.n_batch_episodes:
        replay = []
        print trainer._do_rollout(
                task, rollout_ph, model, desc_model, replay, [], session,
                config, 10000, h0, z0, "val")

        for episode in replay:
            #for experience in episode:
            for experience in episode[1:2]:
            #for experience in episode[1:]:
                codes.append(experience.m1[1][0])
                states.append(experience.s1)
                desc = experience.s1.desc[1]
                #for i_bigram in range(max(1, len(desc) - 1)):
                #    descs.append(tuple(desc[i_bigram:i_bigram+2]))
                #i_bigram = np.random.randint(max(1, len(desc) - 1))
                #descs.append(tuple(desc[i_bigram:i_bigram+2]))
                descs.append(tuple(desc))
        i_rollout += 1

    print len(codes), len(states), len(descs)

    descs = set(descs)

    vis_data = {
        "descs": [],
        "codes": [],
        "states": [],
        "distractors": []
    }

    xb = np.zeros((config.trainer.n_batch_episodes, task.n_features))
    xa_true = np.zeros((config.trainer.n_batch_episodes, task.n_features))
    xa_noise = np.zeros(
            (config.trainer.n_batch_episodes, config.trainer.n_distractors,
                task.n_features))
    for i, state in enumerate(states):
        xb[i, :] = state.obs()[1]
        xa_true[i, :] = state.obs()[0]
        distractors = task.distractors_for(state, 1, config.trainer.n_distractors)
        vis_data["states"].append(task.visualize(state, 0))
        distractor_vis = []
        for i_dis in range(len(distractors)):
            dis, _ = distractors[i_dis]
            xa_noise[i, i_dis, :] = dis.obs()[0]
            distractor_vis.append(task.visualize(dis, 0))
        vis_data["distractors"].append(distractor_vis)

    all_desc_strs = []
    all_desc_beliefs = []
    all_desc_weights = []
    for desc in descs:
    #for word in range(len(task.vocab)):
    #for true_word, _ in task.freq_vocab[:100]:
    #    word = task.vocab[true_word]
    #    desc = [word]
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

        win_count = 0
        for row in desc_beliefs:
            if row[0] > row[1]:
                win_count += 1
        #print "win", win_count

        vis_desc_beliefs = desc_beliefs * desc_weights[:, np.newaxis]
        #vis_desc_beliefs -= np.min(vis_desc_beliefs)
        if np.max(vis_desc_beliefs) > 0:
            vis_desc_beliefs /= np.max(vis_desc_beliefs)
        str_desc = task.pp(desc)
        all_desc_strs.append(str_desc)
        all_desc_beliefs.append(desc_beliefs)
        all_desc_weights.append(np.tile(
                desc_weights[:, np.newaxis],
                (1, config.trainer.n_distractors + 1)))
        vis_data["descs"].append({
            "value": str_desc, 
            "repr": vis_desc_beliefs.tolist(),
        })

    def kl(d1, d2, w1, w2):
        tot = 0
        for p1, p2, pw1, pw2 in zip(d1.ravel(), d2.ravel(), w1.ravel(), w2.ravel()):
            tot += pw1 * pw2 * p1 * (np.log(p1) - np.log(p2))
        return tot
    
    def tvd_global(d1, d2, w1, w2):
        tot = 1
        for p1, p2 in zip(d1.ravel(), d2.ravel()):
            tot = min(tot, np.abs(p1 - p2))
        return tot

    def tvd_local(d1, d2, w1, w2):
        tot = 0
        for i in range(d1.shape[0]):
            here = 1
            for j in range(d1.shape[1]):
                p1 = d1[i, j]
                p2 = d2[i, j]
                here = min(here, np.abs(p1 - p2))
            pw1 = w1[i, 0]
            pw2 = w2[i, 0]
            tot += pw1 * pw2 * here
        return tot

    for code in codes:
        code_data = [code] * config.trainer.n_batch_episodes
        feed = {
            reconst_ph.t_xb: xb,
            reconst_ph.t_xa_true: xa_true,
            reconst_ph.t_xa_noise: xa_noise,
            reconst_ph.t_z: code_data
        }
        model_beliefs, model_weights = session.run(
                [translator.t_model_belief, translator.t_model_weights], feed)
        vis_model_beliefs = model_beliefs * model_weights[:, np.newaxis]
        #vis_model_beliefs -= np.min(vis_model_beliefs)
        if np.max(vis_model_beliefs) > 0:
            vis_model_beliefs /= np.max(vis_model_beliefs)
        simple_code = code.tolist()
        r_model_weights = np.tile(
                model_weights[:, np.newaxis],
                (1, config.trainer.n_distractors + 1))

        by_fkl = sorted(
                range(len(all_desc_strs)), 
                key=lambda i: kl(all_desc_beliefs[i], model_beliefs,
                    all_desc_weights[i], r_model_weights))
        by_rkl = sorted(
                range(len(all_desc_strs)),
                key=lambda i: kl(model_beliefs, all_desc_beliefs[i],
                    r_model_weights, all_desc_weights[i]))

        tvd_func = tvd_global if config.translator.normalization == "global" else tvd_local
        by_tvd = sorted(
                range(len(all_desc_strs)),
                key=lambda i: tvd_func(all_desc_beliefs[i], model_beliefs,
                    all_desc_weights[i], r_model_weights))

        vis_data["codes"].append({
            "value": simple_code, 
            "repr": vis_model_beliefs.tolist(),
            "fkl": "[" + ", ".join([all_desc_strs[i] for i in by_fkl[:3]]) + "]",
            "rkl": "[" + ", ".join([all_desc_strs[i] for i in by_rkl[:3]]) + "]",
            "tvd": "[" + ", ".join([all_desc_strs[i] for i in by_tvd[:3]]) + "]",
        })

    with open(config.experiment_dir + "/vis.json", "w") as vis_f:
        json.dump(vis_data, vis_f)
