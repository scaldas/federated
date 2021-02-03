import functools
import os
import tensorflow as tf

from federated_dropout.utils import keras_metrics
from utils.datasets import shakespeare_dataset

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


def get_dataset(cache_path, sequence_length=80):
  os.makedirs(cache_path, exist_ok=True)

  client_epochs_per_round = 1
  client_batch_size = 4
  max_batches_per_client = -1
  max_test_batches = None

  train_data = shakespeare_dataset.construct_character_level_datasets(
      client_batch_size=client_batch_size,
      client_epochs_per_round=client_epochs_per_round,
      sequence_length=sequence_length,
      max_batches_per_client=max_batches_per_client,
      cache_dir=cache_path)

  _, test_data = shakespeare_dataset.get_centralized_datasets(
      train_batch_size=client_batch_size,
      max_test_batches=max_test_batches,
      sequence_length=sequence_length,
      cache_dir=cache_path)

  return train_data, test_data


def create_model(
  sequence_length=80,
  mask_zero=True,
  vocab_size=VOCAB_SIZE,
  num_lstms=1, # TODO: Change to 2!
  lstm_units=256):
  
  # Tensor 0 corresponds to the embeddings. 
  # Following tensors (in groups of 3s) correspond to the LSTMs:
  #   If u is the number of units and d is the input to the cell,
  #   each LSTM has a kernel (d x 4u), a recurrent kernel (u x 4u) and bias (4u). 
  # Tensors -1 and -2 correspond to the final dense layer. 
  
  model = tf.keras.Sequential()
  model.add(
      tf.keras.layers.Embedding(
          input_dim=vocab_size,
          input_length=sequence_length,
          output_dim=8,
          mask_zero=mask_zero))
  lstm_layer_builder = functools.partial(
      tf.keras.layers.LSTM,
      units=lstm_units,
      kernel_initializer='he_normal',
      return_sequences=True,
      stateful=False)
  for _ in range(num_lstms):
    model.add(lstm_layer_builder())
  model.add(tf.keras.layers.Dense(vocab_size))  # Note: logits, no softmax.
  return model


@tf.function
def map_server_to_client_model(server_weights, client_weights, seed_1, seed_2):  
  # Get activations to keep.
  server_units = server_weights[2].shape[0]
  client_units = client_weights[2].shape[0]
  activations, repeated_activations = get_activations(
    server_units, client_units, seed_1, seed_2)

  #import sys
  #tf.print('Papaya', activations, output_stream=sys.stdout)
  
  embeddings = tf.identity(server_weights[0])

  lstm_kernel = tf.gather(server_weights[1], repeated_activations, axis=1)
  lstm_recurrent = tf.gather(server_weights[2], activations, axis=0)
  lstm_recurrent = tf.gather(lstm_recurrent, repeated_activations, axis=1)
  lstm_bias = tf.gather(server_weights[3], repeated_activations, axis=0)
  
  dense_kernel = tf.gather(server_weights[4], activations, axis=0)  
  dense_bias = tf.identity(server_weights[5])
  
  return [
    embeddings,
    lstm_kernel,
    lstm_recurrent,
    lstm_bias,
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

  # Expand the lstm kernel appropriately.
  target_shape = server_weights[1].shape
  indices = tf.cast(tf.where(tf.ones(target_shape)), tf.int32)
  indices = tf.reshape(indices, [-1, 4 * server_units, len(target_shape)])
  indices = tf.gather(indices, repeated_activations, axis=1)
  lstm_kernel = tf.scatter_nd(indices, client_weights[1], shape=target_shape)

  # Expand the lstm recurrent kernel appropriately.
  target_shape = server_weights[2].shape
  indices = tf.cast(tf.where(tf.ones(target_shape)), tf.int32)
  indices = tf.reshape(indices, [server_units, 4 * server_units, len(target_shape)])
  indices = tf.gather(indices, activations, axis=0)
  indices = tf.gather(indices, repeated_activations, axis=1)
  lstm_recurrent = tf.scatter_nd(indices, client_weights[2], shape=target_shape)

  # Expand the lstm bias appropriately.
  target_shape = server_weights[3].shape
  indices = tf.cast(tf.where(tf.ones(target_shape)), tf.int32)
  indices = tf.reshape(indices, [4 * server_units, len(target_shape)])
  indices = tf.gather(indices, repeated_activations, axis=0)
  lstm_bias = tf.scatter_nd(indices, client_weights[3], shape=target_shape)

  # Expand the dense kernel appropriately.
  target_shape = server_weights[4].shape
  indices = tf.cast(tf.where(tf.ones(target_shape)), tf.int32)
  indices = tf.reshape(indices, [server_units, -1, len(target_shape)])
  indices = tf.gather(indices, activations, axis=0)
  dense_kernel = tf.scatter_nd(indices, client_weights[4], shape=target_shape)

  dense_bias = tf.identity(client_weights[5])

  return [
    embeddings,
    lstm_kernel,
    lstm_recurrent,
    lstm_bias,
    dense_kernel,
    dense_bias
  ]

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
  repeated_activations = tf.concat(
    [
      activations,
      activations + server_units,
      activations + 2 * server_units,
      activations + 3 * server_units
    ],
    axis=0)

  return activations, repeated_activations

