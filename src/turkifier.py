from tasks.ref import RefTask

import numpy as np
import tensorflow as tf
import os

def _sample_ref(
        task, rollout_ph, model, code_to_desc, session, config, h0, z0, fold):
    state = task.get_instance(fold)
    hs, zs = h0, z0
    dhs = h0
    hs_, zs_, qs = session.run(
            [model.tt_rollout_h, model.tt_rollout_z, model.tt_rollout_q],
            rollout_ph.feed(hs, zs, dhs, [state], task, config))
    code = zs_[0][0, :]
    l_descs = code_to_desc(code)[:5]
    l_ens = [" ".join(task.reverse_vocab[w] for w in d) for d in l_descs]
    l_en = "; ".join(l_ens)
    return state, l_en, state.target

def run(task, rollout_ph, model, lexicographer, session, config):
    if not isinstance(task, RefTask):
        return
    h0, z0, _ = session.run(model.zero_state(1, tf.float32))
    count = config.evaluator.n_episodes
    out = []

    turk_dir = config.experiment_dir + "/turk_files"
    os.mkdir(turk_dir)

    for _ in range(count):
        state, l_msg, correct_action = _sample_ref(
                task, rollout_ph, model, lexicographer.c_to_l, session, config,
                h0, z0, "test")

        tag1, tag2 = task.turk_visualize(state, 0, turk_dir)
        out.append((tag1, tag2, l_msg, str(correct_action)))

    with open(config.experiment_dir + "/turk.csv", "w") as turk_f:
        print >>turk_f, "tag1,tag2,msg,target"
        for row in out:
            print >>turk_f, '%s,%s,%s,%s' % row
