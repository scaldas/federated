# Copyright 2020, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simple FedAvg to train EMNIST.

This is intended to be a minimal stand-alone experiment script built on top of
core TFF.
"""

import collections
import functools
from absl import app
from absl import flags
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

from federated_dropout import emnist_models
from federated_dropout import simple_fedavg_tf
from federated_dropout import simple_fedavg_tff

# Training hyperparameters
flags.DEFINE_integer('total_rounds', 256, 'Number of total training rounds.')
flags.DEFINE_integer('rounds_per_eval', 1, 'How often to evaluate')
flags.DEFINE_integer('train_clients_per_round', 2,
                     'How many clients to sample per round.')
flags.DEFINE_integer('client_epochs_per_round', 1,
                     'Number of epochs in the client to take per round.')
flags.DEFINE_integer('batch_size', 20, 'Batch size used on the client.')
flags.DEFINE_integer('test_batch_size', 100, 'Minibatch size of test data.')

# Optimizer configuration (this defines one or more flags per optimizer).
flags.DEFINE_float('server_learning_rate', 1.0, 'Server learning rate.')
flags.DEFINE_float('client_learning_rate', 0.1, 'Client learning rate.')

# Dropout flags.
flags.DEFINE_integer('dropout_seed', 931231, 'Seed to control dropout randomness.')
flags.DEFINE_integer('server_hidden_units', 150, 'Number of hidden units on the server.')
flags.DEFINE_integer('client_hidden_units', 50, 'Number of hidden units on the clients.')

# Other flags.
flags.DEFINE_string(
  'results_path', './results/results.csv', 'Path to save results csv.')
flags.DEFINE_string(
  'cache_path', './cache/', 'Path to cache the dataset.')


FLAGS = flags.FLAGS


def get_emnist_dataset():
  """Loads and preprocesses the EMNIST dataset.

  Returns:
    A `(emnist_train, emnist_test)` tuple where `emnist_train` is a
    `tff.simulation.ClientData` object representing the training data and
    `emnist_test` is a single `tf.data.Dataset` representing the test data of
    all clients.
  """
  os.makedirs(FLAGS.cache_path, exist_ok=True)
  emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
      only_digits=True, cache_dir=FLAGS.cache_path)

  def element_fn(element):
    return collections.OrderedDict(
        x=tf.expand_dims(element['pixels'], -1), y=element['label'])

  def preprocess_train_dataset(dataset):
    # Use buffer_size same as the maximum client dataset size,
    # 418 for Federated EMNIST
    return dataset.map(element_fn).shuffle(buffer_size=418).repeat(
        count=FLAGS.client_epochs_per_round).batch(
            FLAGS.batch_size, drop_remainder=False)

  def preprocess_test_dataset(dataset):
    return dataset.map(element_fn).batch(
        FLAGS.test_batch_size, drop_remainder=False)

  emnist_train = emnist_train.preprocess(preprocess_train_dataset)
  emnist_test = preprocess_test_dataset(
      emnist_test.create_tf_dataset_from_all_clients())
  return emnist_train, emnist_test


def server_optimizer_fn():
  return tf.keras.optimizers.SGD(learning_rate=FLAGS.server_learning_rate)


def client_optimizer_fn():
  return tf.keras.optimizers.SGD(learning_rate=FLAGS.client_learning_rate)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_data, test_data = get_emnist_dataset()

  """These functions construct fully initialized models for use in federated averaging."""
  def tff_server_model_fn():
    server_model = emnist_models.create_fully_connected_model(
      num_hidden_units=FLAGS.server_hidden_units, only_digits=True)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return simple_fedavg_tf.KerasModelWrapper(server_model, test_data.element_spec, loss)

  def tff_client_model_fn():
    client_model = emnist_models.create_fully_connected_model(
      num_hidden_units=FLAGS.client_hidden_units, only_digits=True)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return simple_fedavg_tf.KerasModelWrapper(client_model, test_data.element_spec, loss)

  iterative_process = simple_fedavg_tff.build_federated_averaging_process(
      tff_server_model_fn,
      tff_client_model_fn,
      emnist_models.map_server_to_client_model,
      emnist_models.map_client_to_server_model,
      FLAGS.dropout_seed,
      server_optimizer_fn,
      client_optimizer_fn)
  server_state = iterative_process.initialize()

  metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
  model = tff_server_model_fn() # The evaluation is done with the full server model.
  results_dict = collections.OrderedDict()
  for round_num in range(FLAGS.total_rounds):
    sampled_clients = np.random.choice(
        train_data.client_ids,
        size=FLAGS.train_clients_per_round,
        replace=False)
    sampled_train_data = [
        train_data.create_tf_dataset_for_client(client)
        for client in sampled_clients
    ]
    server_state, train_metrics = iterative_process.next(
        server_state, sampled_train_data, round_num)
    print(f'Round {round_num} training loss: {train_metrics}')
    results_dict[round_num] = collections.OrderedDict(
      server_hidden_units=FLAGS.server_hidden_units,
      client_hidden_units=FLAGS.client_hidden_units,
      dropout_seed=FLAGS.dropout_seed,
      train_loss=train_metrics)
    
    if round_num % FLAGS.rounds_per_eval == 0:
      model.from_weights(server_state.model_weights)
      accuracy = simple_fedavg_tf.keras_evaluate(model.keras_model, test_data, metric)
      
      print(f'Round {round_num} validation accuracy: {accuracy * 100.0}')
      results_dict[round_num]['val_accuracy'] = accuracy.numpy()

  save_results(results_dict)


def save_results(results_dict): 
  results_df = collections.OrderedDict()
  
  for r in results_dict:
    results_df[r] = results_dict[r].values()

  results_df = pd.DataFrame.from_dict(
    results_df,
    orient='index',
    columns=[
      'server_hidden_units',
      'client_hidden_units',
      'dropout_seed',
      'train_loss',
      'val_accuracy'])
  results_df = results_df.rename_axis('round_num')

  print(f'Saving results in {FLAGS.results_path}')
  os.makedirs(os.path.dirname(FLAGS.results_path), exist_ok=True)
  results_df.to_csv(FLAGS.results_path)


if __name__ == '__main__':
  app.run(main)