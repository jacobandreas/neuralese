import numpy as np
import json

random = np.random.RandomState(0)
N_PROJ = 18

def run(lex, task, config):
    proj = np.random.randn(N_PROJ, config.channel.n_msg)
    vis_data = {
        "descs": [],
        "codes": [],
        "states": [],
        "distractors": []
    }

    for state, distractors in zip(lex.states, lex.distractors):
        vis_data["states"].append(task.visualize(state, config.lexicographer.c_agent))
        distractor_vis = []
        for i_dis in range(len(distractors)):
            dis, _ = distractors[i_dis]
            distractor_vis.append(task.visualize(dis, config.lexicographer.c_agent))
        vis_data["distractors"].append(distractor_vis)

    for i, l_msg in enumerate(lex.l_msgs):
        belief = lex.l_beliefs[i]
        weights = lex.l_weights[i]
        str_msg = task.pp(l_msg)
        vis_repr = belief * weights[:, np.newaxis]
        if np.max(vis_repr) > 0:
            vis_repr /= np.max(vis_repr)
        vis_data["descs"].append({
            "value": str_msg,
            "repr": vis_repr.tolist(),
        })

    for i, code in enumerate(lex.codes[:10]):
        belief = lex.model_beliefs[i]
        weights = lex.model_weights[i]
        short_code = proj.dot(code).tolist()
        vis_repr = belief * weights[:, np.newaxis]
        if np.max(vis_repr) > 0:
            vis_repr /= np.max(vis_repr)
        trans = [[task.reverse_vocab[i] for i in t] 
                for t in lex.c_to_l(code, config.lexicographer.mode)]
        trans = ", ".join([" ".join(t) for t in trans])
        vis_data["codes"].append({
            "value": short_code,
            "repr": vis_repr.tolist(),
            "trans": trans
        })

    with open(config.experiment_dir + "/vis.json", "w") as vis_f:
        json.dump(vis_data, vis_f)
