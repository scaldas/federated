import functools
import os
import tensorflow as tf

from utils.datasets import emnist_dataset


def get_emnist_dataset(
  client_epochs_per_round,
  client_batch_size,
  max_batches_per_client,
  test_batch_size,
  max_test_batches,
  cache_path,
  seed,
  only_digits=True):

  """Loads and preprocesses the EMNIST dataset."""
  os.makedirs(cache_path, exist_ok=True)

  #client_epochs_per_round = 1
  #client_batch_size = 20
  #max_batches_per_client = -1
  #max_test_batches = None
  #test_batch_size = 100

  emnist_train, _ = emnist_dataset.get_emnist_datasets(
      client_batch_size=client_batch_size,
      client_epochs_per_round=client_epochs_per_round,
      max_batches_per_client=max_batches_per_client,
      only_digits=only_digits,
      cache_dir=cache_path,
      seed=seed)

  _, emnist_test = emnist_dataset.get_centralized_datasets(
      train_batch_size=client_batch_size,
      test_batch_size=test_batch_size,
      max_test_batches=max_test_batches,
      only_digits=only_digits,
      shuffle_train=False, # shouldn't actually matter.
      cache_dir=cache_path,
      seed=seed)
  
  return emnist_train, emnist_test


def create_model(hidden_units, seed, only_digits=True):
  """Creates a model with two hidden layers."""
  dense_layer_builder = functools.partial(
      tf.keras.layers.Dense,
      kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
      bias_initializer='zeros')

  model = tf.keras.models.Sequential([
      tf.keras.layers.Reshape(input_shape=(28, 28, 1), target_shape=(28 * 28,)),
      dense_layer_builder(units=hidden_units, activation=tf.nn.relu),
      dense_layer_builder(units=hidden_units, activation=tf.nn.relu),
      dense_layer_builder(
        units=10 if only_digits else 62,
        activation=tf.nn.softmax),
  ])

  return model


@tf.function
def map_server_to_client_model(server_weights, client_weights, seed_1, seed_2):
  
  # Gather the appropriate server weights.
  # Indices 0 and 1 are the first layer and biases.
  activations_1 = get_activations(
    server_weights[0].shape[1],
    client_weights[0].shape[1],
    seed_1,
    seed_2)

  layer_1 = tf.gather(server_weights[0], activations_1, axis=1)
  bias_1 = tf.gather(server_weights[1], activations_1, axis=0)
  
  # Indices 2 and 3 are the second layer and biases.
  activations_2 = get_activations(
    server_weights[2].shape[1],
    client_weights[2].shape[1],
    seed_1 + seed_2,
    seed_2)

  layer_2 = tf.gather(server_weights[2], activations_1, axis=0)
  layer_2= tf.gather(layer_2, activations_2, axis=1)
  bias_2 = tf.gather(server_weights[3], activations_2, axis=0)

  # Indices 4 and 5 are the output layer and biases.
  output_layer = tf.gather(server_weights[4], activations_2, axis=0)
  output_bias = tf.identity(server_weights[5])

  return [
    layer_1,
    bias_1,
    layer_2,
    bias_2,
    output_layer,
    output_bias]


@tf.function
def map_client_to_server_model(client_weights, server_weights, seed_1, seed_2):

  # Expand the client weights appropriately.
  # Indices 0 and 1 are the first layer and biases.  
  activations_1 = get_activations(
    server_weights[0].shape[1],
    client_weights[0].shape[1],
    seed_1,
    seed_2)

  target_shape = server_weights[0].shape
  indices = tf.cast(tf.where(tf.ones(target_shape)), tf.int32)
  indices = tf.reshape(
    indices, [-1, server_weights[0].shape[1], len(target_shape)])
  indices = tf.gather(indices, activations_1, axis=1)
  layer_1 = tf.scatter_nd(indices, client_weights[0], shape=target_shape)

  target_shape = server_weights[1].shape
  indices = tf.cast(tf.where(tf.ones(target_shape)), tf.int32)
  indices = tf.reshape(
    indices, [server_weights[1].shape[0], len(target_shape)])
  indices = tf.gather(indices, activations_1, axis=0) 
  bias_1 = tf.scatter_nd(indices, client_weights[1], shape=target_shape)

  # Indices 2 and 3 are the second layer and biases.  
  activations_2 = get_activations(
    server_weights[2].shape[1],
    client_weights[2].shape[1],
    seed_1,
    seed_2)
  
  target_shape = server_weights[2].shape
  indices = tf.cast(tf.where(tf.ones(target_shape)), tf.int32)
  indices = tf.reshape(
    indices, [server_weights[2].shape[0], server_weights[2].shape[1], len(target_shape)])
  indices = tf.gather(indices, activations_1, axis=0)
  indices = tf.gather(indices, activations_2, axis=1)
  layer_2 = tf.scatter_nd(indices, client_weights[2], shape=target_shape)

  target_shape = server_weights[3].shape
  indices = tf.cast(tf.where(tf.ones(target_shape)), tf.int32)
  indices = tf.reshape(
    indices, [server_weights[3].shape[0], len(target_shape)])
  indices = tf.gather(indices, activations_2, axis=0) 
  bias_2 = tf.scatter_nd(indices, client_weights[3], shape=target_shape)

  # Indices 4 and 5 are the output layer and biases.
  target_shape = server_weights[4].shape
  indices = tf.cast(tf.where(tf.ones(target_shape)), tf.int32)
  indices = tf.reshape(
    indices, [server_weights[4].shape[0], -1, len(target_shape)])
  indices = tf.gather(indices, activations_2, axis=0) 
  output_layer = tf.scatter_nd(indices, client_weights[4], shape=target_shape)

  output_bias = tf.identity(client_weights[5])

  return [
    layer_1,
    bias_1,
    layer_2,
    bias_2,
    output_layer,
    output_bias]


@tf.function
def get_activations(server_units, client_units, seed_1, seed_2):
  activations = tf.random.stateless_uniform(
    [server_units], seed=(seed_1, seed_2), minval=0, maxval=1, dtype=tf.float32)
  _, activations = tf.math.top_k(activations, k=client_units, sorted=False)
  activations = tf.sort(activations, direction='ASCENDING')
  return activations
