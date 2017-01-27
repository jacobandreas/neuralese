import tensorflow as tf

RELU_SCALE = 1.43

def mlp(t_in, widths, final_nonlinearity=False):
    weights = []
    prev_width = t_in.get_shape()[-1]
    prev_layer = t_in
    for i_layer, width in enumerate(widths):
        v_w = tf.get_variable("w%d" % i_layer, shape=(prev_width, width),
                initializer=tf.uniform_unit_scaling_initializer(
                    factor=RELU_SCALE))
        v_b = tf.get_variable("b%d" % i_layer, shape=(width,),
                initializer=tf.constant_initializer(0.0))
        weights += [v_w, v_b]

        t_layer = batch_matmul(prev_layer, v_w) + v_b
        if final_nonlinearity or i_layer < len(widths) - 1:
            #t_layer = tf.nn.relu(t_layer)
            t_layer = tf.nn.tanh(t_layer)
        prev_layer = t_layer
        prev_width = width
    return prev_layer, weights

def embed(t_in, n_embeddings, size, multi=False):
    if multi:
        varz = [tf.get_variable("embed%d" % i, shape=(n_embeddings, size),
                    initializer=tf.uniform_unit_scaling_initializer())
                for i in range(t_in.get_shape()[1])]
    else:
        varz = [tf.get_variable("embed", shape=(n_embeddings, size),
                    initializer=tf.uniform_unit_scaling_initializer())]
    embedded = tf.nn.embedding_lookup(varz, t_in)
    eshape = embedded.get_shape()
    if multi:
        embedded = tf.reshape(embedded, (-1, eshape[1].value * eshape[2].value))
    return embedded, varz

# https://github.com/tensorflow/tensorflow/issues/216
def batch_matmul(x, m):
  [input_size, output_size] = m.get_shape().as_list()

  input_shape = tf.shape(x)
  batch_rank = input_shape.get_shape()[0].value - 1
  batch_shape = input_shape[:batch_rank]
  output_shape = tf.concat(0, [batch_shape, [output_size]])

  x = tf.reshape(x, [-1, input_size])
  y = tf.matmul(x, m)

  y = tf.reshape(y, output_shape)

  return y
