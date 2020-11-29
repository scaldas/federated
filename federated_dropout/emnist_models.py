import tensorflow as tf


def create_fully_connected_model(num_hidden_units, only_digits=True):
  """Creates a model with a single hidden layer.

  Args:
    num_hidden_units: Number of units in the hidden layer.
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=[28, 28, 1]),
      tf.keras.layers.Dense(num_hidden_units, activation=tf.nn.relu),
      tf.keras.layers.Dense(10 if only_digits else 62, activation=tf.nn.softmax)
  ])
  return model


@tf.function
def map_server_to_client_model(server_weights, client_weights, seed_1, seed_2):
  # Get activations to keep.
  server_hidden_units = server_weights[0].shape[1]
  client_hidden_units = client_weights[0].shape[1]
  activations = get_activations(
    server_hidden_units, client_hidden_units, seed_1, seed_2)

  # Gather the appropriate server weights.
  hidden_layer = tf.gather(server_weights[0], activations, axis=1)
  hidden_bias = tf.gather(server_weights[1], activations, axis=0)
  output_layer = tf.gather(server_weights[2], activations, axis=0)

  output_bias = tf.identity(server_weights[3])

  return [hidden_layer, hidden_bias, output_layer, output_bias]


@tf.function
def map_client_to_server_model(client_weights, server_weights, seed_1, seed_2):

  # Get activations to keep.
  server_hidden_units = server_weights[0].shape[1]
  client_hidden_units = client_weights[0].shape[1]
  activations = get_activations(
    server_hidden_units, client_hidden_units, seed_1, seed_2)

  # Expand the client weights appropriately.
  target_shape = server_weights[0].shape
  indices = tf.ones(target_shape, dtype=tf.int32)
  indices = tf.cast(tf.where(tf.ones(target_shape)), tf.int32)
  indices = tf.reshape(indices, [-1, server_hidden_units, len(target_shape)])
  indices = tf.gather(indices, activations, axis=1)
  hidden_layer = tf.scatter_nd(indices, client_weights[0], shape=target_shape)

  target_shape = server_weights[1].shape
  indices = tf.cast(tf.where(tf.ones(target_shape)), tf.int32)
  indices = tf.reshape(indices, [server_hidden_units, len(target_shape)])
  indices = tf.gather(indices, activations, axis=0) 
  hidden_bias = tf.scatter_nd(indices, client_weights[1], shape=target_shape)

  target_shape = server_weights[2].shape
  indices = tf.cast(tf.where(tf.ones(target_shape)), tf.int32)
  indices = tf.reshape(indices, [server_hidden_units, -1, len(target_shape)])
  indices = tf.gather(indices, activations, axis=0) 
  output_layer = tf.scatter_nd(indices, client_weights[2], shape=target_shape)

  output_bias = tf.identity(client_weights[3])

  return [hidden_layer, hidden_bias, output_layer, output_bias]


@tf.function
def get_activations(server_units, client_units, seed_1, seed_2):
  activations = tf.random.stateless_uniform(
    [server_units], seed=(seed_1, seed_2), minval=0, maxval=1, dtype=tf.float32)
  _, activations = tf.math.top_k(activations, k=client_units, sorted=False)
  activations = tf.sort(activations, direction='ASCENDING')
  return activations
