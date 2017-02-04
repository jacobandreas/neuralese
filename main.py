#!/usr/bin/env python2

import experience
from util import Struct

import tasks
import models
import models.desc_im
import channels
import translators

import trainer
import lexicographer
import visualizer
import calibrator
import evaluator
import sem_evaluator
import turkifier

import logging
import numpy as np
import os
import sys
import tensorflow as tf
import traceback
import yaml

def main():
    config = configure()
    session = tf.Session()
    task = tasks.load(config)
    channel = channels.load(config)
    model = models.load(config)
    desc_model = models.desc_im.DescriptionImitationModel()
    translator = translators.load(config)

    rollout_ph = experience.RolloutPlaceholders(task, config)
    replay_ph = experience.ReplayPlaceholders(task, config)
    reconst_ph = experience.ReconstructionPlaceholders(task, config)
    channel.build(config)
    model.build(task, rollout_ph, replay_ph, channel, config)
    desc_model.build(task, rollout_ph, replay_ph, channel, config)
    translator.build(task, reconst_ph, channel, model, config)

    if config.task.train:
        trainer.run(
                task, rollout_ph, replay_ph, reconst_ph, model, desc_model,
                translator, session, config)
    else:
        trainer.load(session, config)

    if config.task.lexicon:
        lex = lexicographer.run(
                task, rollout_ph, reconst_ph, model, desc_model, translator,
                session, config)

    if config.task.visualize:
        visualizer.run(lex, task, config)

    if config.task.calibrate:
        calibrator.run(
                task, rollout_ph, model, desc_model, lexicographer, session,
                config)

    if config.task.evaluate:
        evaluator.run(
                task, rollout_ph, replay_ph, reconst_ph, model, desc_model, lex,
                session, config)
        sem_evaluator.run(
                task, rollout_ph, reconst_ph, model, desc_model, translator,
                lex, session, config)

    if config.task.turkify:
        turkifier.run(task, rollout_ph, model, lex, session, config)

def configure():
    tf.set_random_seed(0)

    # load config
    with open(sys.argv[1]) as config_f:
        config = Struct(**yaml.load(config_f))

    # set up experiment
    config.experiment_dir = os.path.join("experiments/%s" % config.name)
    assert not os.path.exists(config.experiment_dir), \
            "Experiment %s already exists!" % config.experiment_dir
    os.mkdir(config.experiment_dir)

    # set up logging
    log_name = os.path.join(config.experiment_dir, "run.log")
    logging.basicConfig(filename=log_name, level=logging.DEBUG,
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
            format='%(asctime)s %(levelname)-8s %(message)s')
    def handler(type, value, tb):
        logging.exception("Uncaught exception: %s", str(value))
        logging.exception("\n".join(traceback.format_exception(type, value, tb)))
    sys.excepthook = handler

    logging.info("BEGIN")
    logging.info(str(config))

    return config

if __name__ == "__main__":
    main()
