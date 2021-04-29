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
import random
import tensorflow as tf
import tensorflow_federated as tff

import simple_fedavg_tf
import simple_fedavg_tff

from utils.datasets import emnist_dataset

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

# Model flags.
flags.DEFINE_integer('hidden_units', 50, 'Number of hidden units.')

# Other flags.
flags.DEFINE_string(
	'results_path', './results/results.csv', 'Path to save results csv.')
flags.DEFINE_string(
	'cache_path', './cache/', 'Path to cache the dataset.')
flags.DEFINE_integer('seed', 0, 'Seed for randomization.')
flags.DEFINE_string('initialization_path', './initialization/initialization.model', 'Path to save/load initialization weights.')
flags.DEFINE_bool('only_digits', None, 'Whether to only consider digits.')

FLAGS = flags.FLAGS


def create_model(num_hidden_units, seed):
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
	    num_hidden_units,
	    activation='relu',
	    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
	    bias_initializer='zeros'),
	  tf.keras.layers.Dropout(0.5), # TODO: remove.
	  tf.keras.layers.Dense(
	    10 if FLAGS.only_digits else 62,
	    activation=tf.nn.softmax,
	    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
	    bias_initializer='zeros'),
	])

	return model


def load_initial_model(num_hidden_units):
	if not os.path.exists(FLAGS.initialization_path):
		os.makedirs(os.path.dirname(FLAGS.initialization_path), exist_ok=True)
		model = create_model(num_hidden_units)
		model.save(FLAGS.initialization_path)
	
	return tf.keras.models.load_model(
		FLAGS.initialization_path, compile=False)


def get_emnist_dataset():
	"""Loads and preprocesses the EMNIST dataset.

	Returns:
		A `(emnist_train, emnist_test)` tuple where `emnist_train` is a
		`tff.simulation.ClientData` object representing the training data and
		`emnist_test` is a single `tf.data.Dataset` representing the test data of
		all clients.
	"""
	os.makedirs(FLAGS.cache_path, exist_ok=True)

	emnist_train, _ = emnist_dataset.get_emnist_datasets(
			client_batch_size=FLAGS.batch_size,
			client_epochs_per_round=FLAGS.client_epochs_per_round,
			max_batches_per_client=-1,
			only_digits=FLAGS.only_digits,
			cache_dir=FLAGS.cache_path,
			seed=FLAGS.seed)

	_, emnist_test = emnist_dataset.get_centralized_datasets(
			train_batch_size=FLAGS.batch_size, # shouldn't actually matter.
			test_batch_size=FLAGS.test_batch_size,
			max_test_batches=None,
			only_digits=FLAGS.only_digits,
			shuffle_train=False, # shouldn't actually matter.
			cache_dir=FLAGS.cache_path,
			seed=FLAGS.seed)
				
	return emnist_train, emnist_test


def server_optimizer_fn():
	return tf.keras.optimizers.SGD(learning_rate=FLAGS.server_learning_rate)


def client_optimizer_fn():
	return tf.keras.optimizers.SGD(learning_rate=FLAGS.client_learning_rate)


def main(argv):

	if len(argv) > 1:
		raise app.UsageError('Too many command-line arguments.')

	np.random.seed(FLAGS.seed)
	os.environ['PYTHONHASHSEED'] = str(FLAGS.seed)
	random.seed(FLAGS.seed)
	tf.compat.v1.set_random_seed(FLAGS.seed)

	train_data, test_data = get_emnist_dataset()

	def tff_model_fn(initial_model=False):
		"""Constructs a fully initialized model for use in federated averaging."""
		model_fn = load_initial_model if initial_model else create_model
		keras_model = model_fn(FLAGS.hidden_units, seed=FLAGS.seed)
		loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
		return simple_fedavg_tf.KerasModelWrapper(
			keras_model, test_data.element_spec, loss)

	iterative_process = simple_fedavg_tff.build_federated_averaging_process(
			tff_model_fn, server_optimizer_fn, client_optimizer_fn)
	
	server_state = iterative_process.initialize()
	metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
	model, results_dict = tff_model_fn(), collections.OrderedDict()
	results_dict[-1] = collections.OrderedDict(
		hidden_units=FLAGS.hidden_units,
		train_loss=None,
		val_accuracy=None,
		clients=None,
		client_samples=None,
		model_weights=[i.numpy().tolist() for i in model.keras_model.trainable_variables])
	
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
				server_state, sampled_train_data)
		results_dict[round_num] = collections.OrderedDict(
			hidden_units=FLAGS.hidden_units, train_loss=train_metrics)
		
		if round_num % FLAGS.rounds_per_eval == 0:
			model.from_weights(server_state.model_weights)
			accuracy = simple_fedavg_tf.keras_evaluate(
				model.keras_model, test_data, metric)
			results_dict[round_num]['val_accuracy'] = accuracy.numpy()
		
			# additional information. TODO: delete
			results_dict[round_num]['clients'] = sampled_clients
			results_dict[round_num]['client_samples'] = [[batch[1].numpy().tolist() for batch in iter(client)] for client in sampled_train_data]
			results_dict[round_num]['model_weights'] = [i.tolist() for i in server_state.model_weights.trainable]

			print(f'Round {round_num} training loss: {train_metrics}')
			print(f'Round {round_num} validation accuracy: {accuracy * 100.0}')
			

	save_results(results_dict)


def save_results(results_dict): 
	results_df = collections.OrderedDict()
	
	for r in results_dict:
		results_df[r] = results_dict[r].values()

	results_df = pd.DataFrame.from_dict(
		results_df,
		orient='index', 
		columns=['hidden_units', 'train_loss', 'val_accuracy', 'clients', 'client_samples', 'model_weights'])
	results_df = results_df.rename_axis('round_num')

	print(f'Saving results in {FLAGS.results_path}')
	os.makedirs(os.path.dirname(FLAGS.results_path), exist_ok=True)
	results_df.to_csv(FLAGS.results_path)


if __name__ == '__main__':
	app.run(main)
