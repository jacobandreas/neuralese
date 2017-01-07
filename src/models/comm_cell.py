import net

import tensorflow as tf

class CommCell(tf.nn.rnn_cell.RNNCell):
    def __init__(
            self, n_agents, n_hidden, n_comm, n_out, channel, communicate,
            symmetric):
        self.n_agents = n_agents
        self.n_hidden = n_hidden
        self.n_comm = n_comm
        self.n_out = n_out if isinstance(n_out, tuple) else (n_out,) * n_agents
        self.communicate = communicate
        self.symmetric = symmetric
        self.channel = channel

        self.agent_scopes = []
        if symmetric:
            for i_agent in range(n_agents):
                self.agent_scopes.append("agent")
            self.reuse_agent_scope = True
        else:
            for i_agent in range(n_agents):
                self.agent_scopes.append("agent_%d" % i_agent)
            self.reuse_agent_scope = False

    @property
    def state_size(self):
        return ((self.n_hidden,) * self.n_agents, 
                (self.n_comm,) * self.n_agents,
                self.n_out)

    @property
    def output_size(self):
        return self.state_size

    def __call__(self, inputs, state, scope=None):
        assert len(inputs) == self.n_agents
        assert len(state) == 3
        assert all(len(s) == self.n_agents for s in state)
        states, comms, _ = state

        with tf.variable_scope(scope or "comm_cell"):
            features = []
            if self.communicate:
                base_cell = tf.nn.rnn_cell.GRUCell(self.n_hidden)
                for i_agent in range(self.n_agents):
                    other_comms = tuple(
                            comms[_ia] for _ia in range(self.n_agents)
                            if _ia != i_agent)
                    features.append(
                            tf.concat(1, (inputs[i_agent],) + other_comms))
            else:
                # for fair comparison, let non-communicating agents use the
                # extra channel capacity for themselves
                #base_cell = tf.nn.rnn_cell.GRUCell(
                #        self.n_hidden + self.n_comm * (self.n_agents - 1))
                base_cell = tf.nn.rnn_cell.GRUCell(self.n_hidden)
                for i_agent in range(self.n_agents):
                    features.append(inputs[i_agent])

            next_states = []
            next_comms = []
            next_outs = []
            for i_agent in range(self.n_agents):
                with tf.variable_scope(self.agent_scopes[i_agent],
                        reuse=(i_agent > 0 and self.reuse_agent_scope)):
                    hidden, _ = net.mlp(
                            features[i_agent], (self.n_hidden,),
                            final_nonlinearity=True)
                    _, next_state = base_cell(hidden, states[i_agent])
                    with tf.variable_scope("comm"):
                        next_comm, _ = net.mlp(next_state, (self.n_comm,))
                    with tf.variable_scope("out"):
                        next_out, _ = net.mlp(
                                next_state, (self.n_out[i_agent],))
                    next_states.append(next_state)
                    next_comms.append(next_comm)
                    next_outs.append(next_out)

            transmitted = [self.channel.transmit(c) for c in next_comms]

        next_state = (tuple(next_states), tuple(transmitted), tuple(next_outs))
        return next_state, next_state
