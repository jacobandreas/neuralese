def build(self, task, rollout_ph, replay_ph, channel, config):
    assert task.n_agents == 2
    assert task.n_actions[0] == 1
    with tf.variable_scope("speaker") as speaker_scope:
        replay_message, v_speaker = net.mlp(
                replay_ph.t_x[0], (config.model.n_hidden, config.channel.n_msg))
        replay_transmitted_msg = channel.transmit(replay_message)

        speaker_scope.reuse_variables()

        rollout_message, v_speaker = net.mlp(
                rollout_ph.t_x[0], (config.model.n_hidden, config.channel.n_msg))
