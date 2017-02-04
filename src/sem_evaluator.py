from experience import Experience
import trainer
from tasks.ref import RefTask
from util import Break

import logging
import numpy as np
import tensorflow as tf

random = np.random.RandomState(8935)

def get_l_samples(count, task, fold):
    states = []
    l_msgs = []
    while len(states) < count:
        demo = task.get_demonstration(fold)
        ok_transitions = [t for t in demo if t.s2.l_msg[1][0] != 1]
        if len(ok_transitions) > 0:
            tr = ok_transitions[random.randint(len(ok_transitions))]
            states.append(tr.s1)
            l_msgs.append(tr.s2.l_msg[1])
    return states, l_msgs

def get_c_samples(count, task, rollout_ph, model, desc_model, session, config,
        fold):
    h0, z0, _ = session.run(model.zero_state(1, tf.float32))
    states = []
    c_msgs = []
    try:
        while True:
            replay = []
            rew = trainer._do_rollout(
                    task, rollout_ph, model, desc_model, replay, [], session,
                    config, 10000, h0, z0, fold)

            for episode in replay:
                exp = episode[random.randint(len(episode))]
                c_msgs.append(exp.m2[1][0])
                states.append(exp.s1)
                if len(states) >= count:
                    raise Break()
    except Break:
        pass

    return states, c_msgs

def make_feed_vars(states, task, config):
    xb = np.zeros((config.trainer.n_batch_episodes, task.n_features))
    xa_true = np.zeros((config.trainer.n_batch_episodes, task.n_features))
    xa_noise = np.zeros(
            (config.trainer.n_batch_episodes, config.trainer.n_distractors,
                task.n_features))
    for i, state in enumerate(states):
        xb[i, :] = state.obs()[1]
        xa_true[i, :] = state.obs()[0]
        distractors = task.distractors_for(state, 1, config.trainer.n_distractors)
        for i_dis in range(len(distractors)):
            dis, _ = distractors[i_dis]
            xa_noise[i, i_dis, :] = dis.obs()[0]
    
    return xb, xa_true, xa_noise

def compute_l_beliefs(self, l_data, xb, xa_true, xa_noise, reconst_ph,
        translator, session):
    feed = {
        reconst_ph.t_xb: xb,
        reconst_ph.t_xa_true: xa_true,
        reconst_ph.t_xa_noise: xa_noise,
        reconst_ph.t_l_msg: l_data
    }
    l_beliefs, = session.run([translator.t_desc_belief], feed)
    return l_beliefs

def compute_c_beliefs(self, code_data, xb, xa_true, xa_noise, reconst_ph,
        translator, session):
    feed = {
        reconst_ph.t_xb: xb,
        reconst_ph.t_xa_true: xa_true,
        reconst_ph.t_xa_noise: xa_noise,
        reconst_ph.t_z: code_data
    }
    model_beliefs, = session.run([translator.t_model_belief], feed)
    return model_beliefs

def run(task, rollout_ph, reconst_ph, model, desc_model, translator,
        lexicographer, session, config, fold="test"):
    #if isinstance(task, RefTask):
    #    count = config.evaluator.n_episodes
    #else:
    #    count = 100
    count = config.trainer.n_batch_episodes

    l_states, l_msgs = get_l_samples(count, task, fold)
    c_states, c_msgs = get_c_samples(count, task, rollout_ph, model, desc_model,
            session, config, fold)

    #print l_msgs[:2]
    #print
    #print c_msgs[:2]
    #exit()

    l_xb, l_xa_true, l_xa_noise = make_feed_vars(l_states, task, config)
    c_xb, c_xa_true, c_xa_noise = make_feed_vars(c_states, task, config)

    cc_beliefs = compute_c_beliefs(c_states, c_msgs, c_xb, c_xa_true, c_xa_noise,
            reconst_ph, translator, session)
    ll_beliefs = compute_l_beliefs(l_states, l_msgs, l_xb, l_xa_true,
            l_xa_noise, reconst_ph, translator, session)

    with open(config.experiment_dir + "/sem_eval.txt", "w") as eval_f:
        cc_success = np.mean(np.argmax(cc_beliefs, axis=1) == 0)
        ll_success = np.mean(np.argmax(ll_beliefs, axis=1) == 0)
        logging.info("[l-l]  \t%s" % str(ll_success))
        logging.info("[c-c]  \t%s" % str(cc_success))
        logging.info("")
        print >>eval_f, "(l-l)", ll_success
        print >>eval_f, "(c-c)", cc_success
        print >>eval_f

        for mode in ["fkl", "rkl", "pmi", "dot", "rand"]:
            tr_c = [lexicographer.l_to_c(l, mode)[0] for l in l_msgs]
            tr_l_pre = [lexicographer.c_to_l(c, mode)[:5] for c in c_msgs]
            tr_l = [np.zeros(len(task.lexicon)) for _ in range(len(tr_l_pre))]
            for i in range(len(tr_l)):
                for l_word in tr_l_pre[i]:
                    tr_l[i][task.lexicon.index(l_word)] += 1
                tr_l[i] /= np.sum(tr_l[i])

            tr_c_beliefs = compute_c_beliefs(l_states, tr_c, l_xb, l_xa_true,
                    l_xa_noise, reconst_ph, translator, session)
            tr_l_beliefs = compute_l_beliefs(c_states, tr_l, c_xb, c_xa_true,
                    c_xa_noise, reconst_ph, translator, session)

            c_success = np.mean(np.argmax(tr_c_beliefs, axis=1) == 0)
            l_success = np.mean(np.argmax(tr_l_beliefs, axis=1) == 0)
            
            logging.info("[l-c:%s]\t%s" % (mode, str(c_success)))
            logging.info("[c-l:%s]\t%s" % (mode, str(l_success)))
            logging.info("")
            print >>eval_f, mode + ":"
            print >>eval_f, "(l-c)", c_success
            print >>eval_f, "(c-l)", l_success
            print >>eval_f
