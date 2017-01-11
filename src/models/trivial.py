import tensorflow as tf

class TrivialModel(object):
    def build(self, task, rollout_ph, replay_ph, channel, config):
        self.t_loss = tf.zeros(())
        self.t_train_op = tf.zeros((1,))
        self.oo_update_target = []
        self.tt_rollout_h = [tf.zeros((1, config.model.n_hidden))] * task.n_agents
        self.tt_rollout_z = [tf.zeros((1, config.channel.n_msg))] * task.n_agents
        self.tt_rollout_q = [tf.zeros((1, a)) for a in task.n_actions]
        self.n_actions = task.n_actions
        self.n_hidden = config.model.n_hidden
        self.n_msg = config.channel.n_msg

    def zero_state(self, n, t):
        return (
            [tf.zeros((n, self.n_hidden))] * len(self.n_actions),
            [tf.zeros((n, self.n_msg))] * len(self.n_actions), 
            [tf.zeros((n, a)) for a in self.n_actions])
