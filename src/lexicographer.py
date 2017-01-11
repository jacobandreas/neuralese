from experience import Experience
import trainer

import json
import numpy as np
import tensorflow as tf

class Break(Exception):
    pass

def run(
        task, rollout_ph, replay_ph, reconst_ph, model, translator, session,
        config):
    replay = []
    h0, z0, _ = session.run(model.zero_state(1, tf.float32))
    for _ in range(config.trainer.n_batch_episodes):
        trainer._do_rollout(
                task, rollout_ph, model, replay, [], session, config, 10000, h0, z0)

    codes = []
    states = []
    descs = []
    try:
        for episode in replay:
            #for experience in episode:
            for experience in episode[1:2]:
                codes.append(experience.m1[1][0])
                states.append(experience.s1)
                descs.append(tuple(experience.s1.desc))
                if len(codes) >= config.trainer.n_batch_episodes:
                    raise Break()
    except Break:
        pass

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
        distractors = task.distractors_for(state, config.trainer.n_distractors)
        vis_data["states"].append(task.visualize(state, 0))
        distractor_vis = []
        for i_dis in range(len(distractors)):
            dis, _ = distractors[i_dis]
            xa_noise[i, i_dis, :] = dis.obs()[0]
            distractor_vis.append(task.visualize(dis, 0))
        vis_data["distractors"].append(distractor_vis)

    for desc in descs:
        desc_data = np.zeros(
                (config.trainer.n_batch_episodes, task.max_desc_len))
        desc_data[:, :len(desc)] = desc
        feed = {
            reconst_ph.t_xb: xb,
            reconst_ph.t_xa_true: xa_true,
            reconst_ph.t_xa_noise: xa_noise,
            reconst_ph.t_desc: desc_data
        }
        desc_beliefs, = session.run([translator.t_desc_belief], feed)
        #str_desc = " ".join([task.reverse_vocab[w] for w in desc[1:-1]])
        str_desc = task.pp(desc)
        vis_data["descs"].append(
                {"value": str_desc, "repr": desc_beliefs.tolist()})

    def kl(d1, d2):
        tot = 0
        n1 = np.asarray(d1).ravel()
        n2 = np.asarray(d2).ravel()
        for p1, p2 in zip(n1, n2):
            tot += p1 * (np.log(p1) - np.log(p2))
        return tot

    def tvd(d1, d2):
        tot = 1
        n1 = np.asarray(d1).ravel()
        n2 = np.asarray(d2).ravel()
        for p1, p2 in zip(n1, n2):
            tot = min(tot, np.abs(p1 - p2))
        return tot

    for code in codes:
        code_data = [code] * config.trainer.n_batch_episodes
        feed = {
            reconst_ph.t_xb: xb,
            reconst_ph.t_xa_true: xa_true,
            reconst_ph.t_xa_noise: xa_noise,
            reconst_ph.t_z: code_data
        }
        model_beliefs, = session.run([translator.t_model_belief], feed)
        simple_code = code.tolist()

        by_forward_kl = sorted(vis_data["descs"], key=lambda d: kl(d["repr"],
                model_beliefs))
        by_reverse_kl = sorted(vis_data["descs"], key=lambda d:
                kl(model_beliefs, d["repr"]))
        by_tvd = sorted(vis_data["descs"], key=lambda d: tvd(model_beliefs,
                d["repr"]))

        vis_data["codes"].append({
            "value": simple_code, 
            "repr": model_beliefs.tolist(),
            "fkl": "[" + ", ".join([d["value"] for d in by_forward_kl[:3]]) + "]",
            "rkl": "[" + ", ".join([d["value"] for d in by_reverse_kl[:3]]) + "]",
            "tvd": "[" + ", ".join([d["value"] for d in by_tvd[:3]]) + "]"
        })

    with open(config.experiment_dir + "/vis.json", "w") as vis_f:
        json.dump(vis_data, vis_f)
