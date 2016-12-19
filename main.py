#!/usr/bin/env python2

import tasks.ref
import models.mlp
import models.interp

from collections import namedtuple
import tensorflow as tf

N_BATCH = 256
N_HIDDEN = 256
N_CODE = 8

task = tasks.ref.RefTask()
model = models.mlp.MlpModel(task, N_BATCH, N_HIDDEN, N_CODE)
interpreter_model = models.interp.InterpreterModel(model)
session = tf.Session()
session.run(tf.initialize_all_variables())

i_iter = 0
while True:
    batch = task.get_batch(N_BATCH)
    loss, acc, _ = session.run(
        [model.t_loss, model.t_acc, model.t_train_op],
        {
            model.t_in_target: batch.target,
            model.t_in_distractor: batch.distractor,
            model.t_in_left: batch.left,
            model.t_in_right: batch.right,
            model.t_label: batch.label
        }
    )

    i_iter += 1
    if i_iter % 100 == 0:
        print "%0.4f   %0.4f" % (loss, acc)
