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
  # hidden_units is 128
  data_format = 'channels_last'

  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        activation='relu',
        data_format=data_format,
        input_shape=(28, 28, 1),
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
        bias_initializer='zeros'),
      tf.keras.layers.Conv2D(
        64,
        kernel_size=(3, 3),
        activation='relu',
        data_format=data_format,
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
        bias_initializer='zeros'),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format=data_format),
      tf.keras.layers.Dropout(0.25), # TODO: remove.
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
        hidden_units,
        activation='relu',
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
        bias_initializer='zeros'),
      tf.keras.layers.Dropout(0.5), # TODO: remove.
      tf.keras.layers.Dense(
        10 if only_digits else 62,
        activation=tf.nn.softmax,
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
        bias_initializer='zeros'),
  ])

  return model


@tf.function
def map_server_to_client_model(server_weights, client_weights, seed_1, seed_2):
  # Get activations to keep.
  server_hidden_units = server_weights[4].shape[1]
  client_hidden_units = client_weights[4].shape[1]
  
  activations = get_activations(
    server_hidden_units, client_hidden_units, seed_1, seed_2)

  # Matrix between the flattened MaxPool2D and the Dense layer.
  max_layer = tf.gather(server_weights[4], activations, axis=1)
  max_bias = tf.gather(server_weights[5], activations, axis=0)
  
  # Matrix between the Dense layer and the softmax layer.
  output_layer = tf.gather(server_weights[6], activations, axis=0)

  return [
    tf.identity(server_weights[0]),
    tf.identity(server_weights[1]),
    tf.identity(server_weights[2]),
    tf.identity(server_weights[3]),
    max_layer,
    max_bias,
    output_layer,
    tf.identity(server_weights[7])]


@tf.function
def map_client_to_server_model(client_weights, server_weights, seed_1, seed_2):

  # Get activations to keep.
  server_hidden_units = server_weights[4].shape[1]
  client_hidden_units = client_weights[4].shape[1]
  activations = get_activations(
    server_hidden_units, client_hidden_units, seed_1, seed_2)

  # Matrix between the flattened MaxPool2D and the Dense layer.
  target_shape = server_weights[4].shape
  indices = tf.cast(tf.where(tf.ones(target_shape)), tf.int32)
  indices = tf.reshape(indices, [-1, server_hidden_units, len(target_shape)])
  indices = tf.gather(indices, activations, axis=1)
  max_layer = tf.scatter_nd(indices, client_weights[4], shape=target_shape)

  target_shape = server_weights[5].shape
  indices = tf.cast(tf.where(tf.ones(target_shape)), tf.int32)
  indices = tf.reshape(indices, [server_hidden_units, len(target_shape)])
  indices = tf.gather(indices, activations, axis=0) 
  max_bias = tf.scatter_nd(indices, client_weights[5], shape=target_shape)

  # Matrix between the Dense layer and the softmax layer.
  target_shape = server_weights[6].shape
  indices = tf.cast(tf.where(tf.ones(target_shape)), tf.int32)
  indices = tf.reshape(indices, [server_hidden_units, -1, len(target_shape)])
  indices = tf.gather(indices, activations, axis=0) 
  output_layer = tf.scatter_nd(indices, client_weights[6], shape=target_shape)

  return [
    tf.identity(client_weights[0]),
    tf.identity(client_weights[1]),
    tf.identity(client_weights[2]),
    tf.identity(client_weights[3]),
    max_layer,
    max_bias,
    output_layer,
    tf.identity(client_weights[7])]


@tf.function
def get_activations(server_units, client_units, seed_1, seed_2):
  activations = tf.random.stateless_uniform(
    [server_units], seed=(seed_1, seed_2), minval=0, maxval=1, dtype=tf.float32)
  _, activations = tf.math.top_k(activations, k=client_units, sorted=False)
  activations = tf.sort(activations, direction='ASCENDING')
  return activations
