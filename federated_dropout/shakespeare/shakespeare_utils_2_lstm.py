import functools
import os
import tensorflow as tf

from federated_dropout.utils import keras_metrics
from utils.datasets import shakespeare_dataset

# VOCAB_SIZE is 90.
VOCAB_SIZE = len(shakespeare_dataset.CHAR_VOCAB) + 4


def get_metric():
  """Returns a `list` of `tf.keras.metric.Metric` objects."""
  pad_token, _, _, _ = shakespeare_dataset.get_special_tokens()

  return keras_metrics.MaskedCategoricalAccuracy(masked_tokens=[pad_token])
  #return [
  #    keras_metrics.NumBatchesCounter(),
  #    keras_metrics.NumExamplesCounter(),
  #    keras_metrics.NumTokensCounter(masked_tokens=[pad_token]),
  #    keras_metrics.MaskedCategoricalAccuracy(masked_tokens=[pad_token]),
  #]


def get_dataset(
  cache_path,
  seed,
  sequence_length=80,
  client_epochs_per_round=1,
  client_batch_size=4,
  test_batch_size=100,
  max_batches_per_client=-1,
  max_test_batches=None):
  
  os.makedirs(cache_path, exist_ok=True)

  train_data = shakespeare_dataset.construct_character_level_datasets(
      client_batch_size=client_batch_size,
      client_epochs_per_round=client_epochs_per_round,
      seed=seed,
      sequence_length=sequence_length,
      max_batches_per_client=max_batches_per_client,
      cache_dir=cache_path)

  _, test_data = shakespeare_dataset.get_centralized_datasets(
      train_batch_size=client_batch_size,
      test_batch_size=test_batch_size,
      seed=seed,
      max_test_batches=max_test_batches,
      sequence_length=sequence_length,
      cache_dir=cache_path)

  return train_data, test_data


def create_model(
  seed,
  sequence_length=80,
  mask_zero=True,
  vocab_size=VOCAB_SIZE,
  num_lstms=2,
  lstm_units=256):
  
  # Tensor 0 corresponds to the embeddings. 
  # Following tensors (in groups of 3s) correspond to the LSTMs:
  #   If u is the number of units and d is the input to the cell,
  #   each LSTM has a kernel (d x 4u), a recurrent kernel (u x 4u) and bias (4u). 
  # Tensors -1 and -2 correspond to the final dense layer. 
  
  lstm_layer_builder = functools.partial(
    tf.keras.layers.LSTM,
    units=lstm_units,
    kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
    return_sequences=True,
    stateful=False)

  model = tf.keras.Sequential()
  model.add(
    # I'm not sure if I should seed this.
    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        input_length=sequence_length,
        output_dim=8,
        mask_zero=mask_zero))
  for _ in range(num_lstms):
    model.add(lstm_layer_builder())
  model.add(
    tf.keras.layers.Dense(
      vocab_size,
      activation=None,
      kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
      bias_initializer='zeros'))  # Note: logits, no softmax.
  return model


@tf.function
def map_server_to_client_model(server_weights, client_weights, seed_1, seed_2):  
  # Get activations to keep.
  server_units = server_weights[2].shape[0]
  client_units = client_weights[2].shape[0]
  activations, repeated_activations = get_activations(
    server_units, client_units, seed_1, seed_2)
  
  embeddings = tf.identity(server_weights[0])

  lstm_1_kernel = tf.gather(server_weights[1], repeated_activations, axis=1)
  lstm_1_recurrent = tf.gather(server_weights[2], activations, axis=0)
  lstm_1_recurrent = tf.gather(lstm_1_recurrent, repeated_activations, axis=1)
  lstm_1_bias = tf.gather(server_weights[3], repeated_activations, axis=0)
  
  lstm_2_kernel = tf.gather(server_weights[4], activations, axis=0)
  lstm_2_kernel = tf.gather(lstm_2_kernel, repeated_activations, axis=1)
  lstm_2_recurrent = tf.gather(server_weights[5], activations, axis=0)
  lstm_2_recurrent = tf.gather(lstm_2_recurrent, repeated_activations, axis=1)
  lstm_2_bias = tf.gather(server_weights[6], repeated_activations, axis=0)

  dense_kernel = tf.gather(server_weights[7], activations, axis=0)  
  dense_bias = tf.identity(server_weights[8])
  
  return [
    embeddings,
    lstm_1_kernel,
    lstm_1_recurrent,
    lstm_1_bias,
    lstm_2_kernel,
    lstm_2_recurrent,
    lstm_2_bias,
    dense_kernel,
    dense_bias
  ]


@tf.function
def map_client_to_server_model(client_weights, server_weights, seed_1, seed_2):
  # Get activations to keep.
  server_units = server_weights[2].shape[0]
  client_units = client_weights[2].shape[0]
  activations, repeated_activations = get_activations(
    server_units, client_units, seed_1, seed_2)

  embeddings = tf.identity(client_weights[0])

  # Expand the first lstm cell appropriately.
  target_shape = server_weights[1].shape
  indices = tf.cast(tf.where(tf.ones(target_shape)), tf.int32)
  indices = tf.reshape(indices, [-1, 4 * server_units, len(target_shape)])
  indices = tf.gather(indices, repeated_activations, axis=1)
  lstm_1_kernel = tf.scatter_nd(indices, client_weights[1], shape=target_shape)

  target_shape = server_weights[2].shape
  indices = tf.cast(tf.where(tf.ones(target_shape)), tf.int32)
  indices = tf.reshape(indices, [server_units, 4 * server_units, len(target_shape)])
  indices = tf.gather(indices, activations, axis=0)
  indices = tf.gather(indices, repeated_activations, axis=1)
  lstm_1_recurrent = tf.scatter_nd(indices, client_weights[2], shape=target_shape)

  target_shape = server_weights[3].shape
  indices = tf.cast(tf.where(tf.ones(target_shape)), tf.int32)
  indices = tf.reshape(indices, [4 * server_units, len(target_shape)])
  indices = tf.gather(indices, repeated_activations, axis=0)
  lstm_1_bias = tf.scatter_nd(indices, client_weights[3], shape=target_shape)

  # Expand the second lstm cell appropriately.
  target_shape = server_weights[4].shape
  indices = tf.cast(tf.where(tf.ones(target_shape)), tf.int32)
  indices = tf.reshape(indices, [server_units, 4 * server_units, len(target_shape)])
  indices = tf.gather(indices, activations, axis=0)
  indices = tf.gather(indices, repeated_activations, axis=1)
  lstm_2_kernel = tf.scatter_nd(indices, client_weights[4], shape=target_shape)

  target_shape = server_weights[5].shape
  indices = tf.cast(tf.where(tf.ones(target_shape)), tf.int32)
  indices = tf.reshape(indices, [server_units, 4 * server_units, len(target_shape)])
  indices = tf.gather(indices, activations, axis=0)
  indices = tf.gather(indices, repeated_activations, axis=1)
  lstm_2_recurrent = tf.scatter_nd(indices, client_weights[5], shape=target_shape)

  target_shape = server_weights[6].shape
  indices = tf.cast(tf.where(tf.ones(target_shape)), tf.int32)
  indices = tf.reshape(indices, [4 * server_units, len(target_shape)])
  indices = tf.gather(indices, repeated_activations, axis=0)
  lstm_2_bias = tf.scatter_nd(indices, client_weights[6], shape=target_shape)

  # Expand the dense kernel appropriately.
  target_shape = server_weights[7].shape
  indices = tf.cast(tf.where(tf.ones(target_shape)), tf.int32)
  indices = tf.reshape(indices, [server_units, -1, len(target_shape)])
  indices = tf.gather(indices, activations, axis=0)
  dense_kernel = tf.scatter_nd(indices, client_weights[7], shape=target_shape)

  dense_bias = tf.identity(client_weights[8])

  return [
    embeddings,
    lstm_1_kernel,
    lstm_1_recurrent,
    lstm_1_bias,
    lstm_2_kernel,
    lstm_2_recurrent,
    lstm_2_bias,
    dense_kernel,
    dense_bias
  ]


@tf.function
def get_activations(server_units, client_units, seed_1, seed_2):
  activations = tf.random.stateless_uniform(
    [server_units], seed=(seed_1, seed_2), minval=0, maxval=1, dtype=tf.float32)
  _, activations = tf.math.top_k(activations, k=client_units, sorted=False)
  
  activations = tf.sort(activations, direction='ASCENDING')
  repeated_activations = tf.concat(
    [
      activations,
      activations + server_units,
      activations + 2 * server_units,
      activations + 3 * server_units
    ],
    axis=0)

  return activations, repeated_activations
