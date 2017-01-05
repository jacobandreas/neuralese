#!/usr/bin/env python2

import tasks.dummy_ref
import tasks.image_ref
import tasks.color_ref
import models.mlp
#import models.interp
import translators.belief

from collections import namedtuple
import tensorflow as tf
import numpy as np

N_BATCH = 256
N_HIDDEN = 256
N_CODE = 256

#task = tasks.dummy_ref.DummyRefTask()
#task = tasks.image_ref.ImageRefTask()
task = tasks.color_ref.ColorRefTask()
model = models.mlp.MlpModel(task, N_BATCH, N_HIDDEN, N_CODE)
translator = translators.belief.BeliefTranslator(task, model, N_BATCH, N_HIDDEN, N_CODE)
session = tf.Session()
session.run(tf.initialize_all_variables())

i_iter = 0
while True:
    batch = task.get_batch(N_BATCH)
    m_loss, m_acc, t_loss, _, _, code = session.run(
        [model.t_loss, model.t_acc, translator.t_loss, model.t_train_op,
            translator.t_train_op, model.t_code],
        {
            model.t_in_target: batch.target,
            model.t_in_distractor: batch.distractor,
            model.t_in_left: batch.left,
            model.t_in_right: batch.right,
            translator.t_in_sentence: batch.sentence,
            model.t_label: batch.label
        }
    )

    i_iter += 1
    if i_iter % 100 == 0:
        #embeddings, = session.run([interp_model.t_embedding])
        #embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        #ex_code = code[0, :]
        #scores = np.dot(embeddings, ex_code)
        #nearest = np.argsort(scores)

        print "%0.4f %0.4f | %0.4f" % (m_loss, m_acc, t_loss)
        #print task.decode(batch.sentence[0])
        #print task.decode(nearest[-10:])
        print
