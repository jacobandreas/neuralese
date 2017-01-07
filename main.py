#!/usr/bin/env python2

import experience
from util import Struct

import tasks
import models
import channels
import translators

import trainer
import lexicographer

import logging
import numpy as np
import os
import sys
import tensorflow as tf
import traceback
import yaml

def main():
    config = configure()
    task = tasks.load(config)
    channel = channels.load(config)
    model = models.load(config)
    translator = translators.load(config)

    session = tf.Session()
    rollout_ph = experience.RolloutPlaceholders(task, config)
    replay_ph = experience.ReplayPlaceholders(task, config)
    channel.build(config)
    model.build(task, rollout_ph, replay_ph, channel, config)
    translator.build(task, rollout_ph, replay_ph, channel, model, config)

    if config.experiment.train:
        trainer.run(task, rollout_ph, replay_ph, model, session, config)
    else:
        trainer.load(session, config)

    if config.experiment.translate:
        lexicographer.run()

def configure():
    tf.set_random_seed(0)

    # load config
    with open("config.yaml") as config_f:
        config = Struct(**yaml.load(config_f))

    # set up experiment
    config.experiment_dir = os.path.join("experiments/%s" % config.name)
    assert not os.path.exists(config.experiment_dir), \
            "Experiment %s already exists!" % config.experiment_dir
    os.mkdir(config.experiment_dir)

    # set up logging
    log_name = os.path.join(config.experiment_dir, "run.log")
    #logging.basicConfig(filename=log_name, level=logging.DEBUG,
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
            format='%(asctime)s %(levelname)-8s %(message)s')
    def handler(type, value, tb):
        logging.exception("Uncaught exception: %s", str(value))
        logging.exception("\n".join(traceback.format_exception(type, value, tb)))
    sys.excepthook = handler

    logging.info("BEGIN")
    logging.info(str(config))

    return config

def train():
    pass

def load():
    pass

def translate():
    pass

if __name__ == "__main__":
    main()
