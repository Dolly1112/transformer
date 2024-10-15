# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic helper functions used across codebase."""

import os
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


# Generic.
def get_single_col_by_input_type(input_type, column_definition):
  """Returns name of single column.

  Args:
    input_type: Input type of column to extract
    column_definition: Column definition list for experiment
  """

  l = [tup[0] for tup in column_definition if tup[2] == input_type]

  if len(l) != 1:
    raise ValueError('Invalid number of columns for {}'.format(input_type))

  return l[0]


def extract_cols_from_data_type(data_type, column_definition,
                                excluded_input_types):
  """Extracts the names of columns that correspond to a define data_type.

  Args:
    data_type: DataType of columns to extract.
    column_definition: Column definition to use.
    excluded_input_types: Set of input types to exclude

  Returns:
    List of names for columns with data type specified.
  """
  return [
      tup[0]
      for tup in column_definition
      if tup[1] == data_type and tup[2] not in excluded_input_types
  ]


# Loss functions.
def tensorflow_quantile_loss(y, y_pred, quantile):
  """Computes quantile loss for tensorflow.

  Standard quantile loss as defined in the "Training Procedure" section of
  the main TFT paper

  Args:
    y: Targets
    y_pred: Predictions
    quantile: Quantile to use for loss calculations (between 0 & 1)

  Returns:
    Tensor for quantile loss.
  """

  # Checks quantile
  if quantile < 0 or quantile > 1:
    raise ValueError(
        'Illegal quantile value={}! Values should be between 0 and 1.'.format(
            quantile))

  prediction_underflow = y - y_pred
  q_loss = quantile * tf.maximum(prediction_underflow, 0.) + (
      1. - quantile) * tf.maximum(-prediction_underflow, 0.)

  return tf.reduce_sum(q_loss, axis=-1)


def numpy_normalised_quantile_loss(y, y_pred, quantile):
  """Computes normalised quantile loss for numpy arrays.

  Uses the q-Risk metric as defined in the "Training Procedure" section of the
  main TFT paper.

  Args:
    y: Targets
    y_pred: Predictions
    quantile: Quantile to use for loss calculations (between 0 & 1)

  Returns:
    Float for normalised quantile loss.
  """
  prediction_underflow = y - y_pred
  weighted_errors = quantile * np.maximum(prediction_underflow, 0.) \
      + (1. - quantile) * np.maximum(-prediction_underflow, 0.)

  quantile_loss = weighted_errors.mean()
  normaliser = y.abs().mean()

  return 2 * quantile_loss / normaliser


# OS related functions.
def create_folder_if_not_exist(directory):
  """Creates folder if it doesn't exist.

  Args:
    directory: Folder path to create.
  """
  # Also creates directories recursively
  pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


# Tensorflow related functions.
def get_default_tensorflow_config(tf_device='gpu', gpu_id=0):
  """Creates tensorflow config for graphs to run on CPU or GPU.

  Specifies whether to run graph on gpu or cpu and which GPU ID to use for multi
  GPU machines.

  Args:
    tf_device: 'cpu' or 'gpu'
    gpu_id: GPU ID to use if relevant

  Returns:
    Tensorflow config.
  """

  if tf_device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # for training on cpu
    tf_config = tf.ConfigProto(
        log_device_placement=False, device_count={'GPU': 0})

  else:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print('Selecting GPU ID={}'.format(gpu_id))

    tf_config = tf.ConfigProto(log_device_placement=False)
    tf_config.gpu_options.allow_growth = True

  return tf_config


def save(tf_session, model_folder, cp_name, scope=None):
  """Saves Tensorflow graph to checkpoint.
  Saves all trainiable variables under a given variable scope to checkpoint.
  将 TensorFlow 图和所有可训练变量保存到指定的检查点文件中。函数的设计允许用户选择是否只保存某个变量范围内的变量

  Args:
    tf_session: Session containing graph
    model_folder: Folder to save models
    cp_name: Name of Tensorflow checkpoint
    scope: Variable scope containing variables to save
  """
  # Save model
  if scope is None:
    saver = tf.train.Saver() # 创建一个 tf.train.Saver() 实例，负责保存当前会话中所有可训练变量的状态。这种情况下，所有可训练变量都会被保存。
  else:
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope) # 获取在指定范围内的所有可训练变量。
    saver = tf.train.Saver(var_list=var_list, max_to_keep=100000) # 创建一个 tf.train.Saver 实例，并传入这些变量的列表。max_to_keep=100000 参数指定保存的检查点的最大数量，防止由于检查点数量过多而导致的问题

  save_path = saver.save(tf_session,
                         os.path.join(model_folder, '{0}.ckpt'.format(cp_name))) # 格式为 model_folder/{cp_name}.ckpt
  print('Model saved to: {0}'.format(save_path)) # 打印保存的检查点路径，以便用户确认模型已成功保存


def load(tf_session, model_folder, cp_name, scope=None, verbose=False):
  """Loads Tensorflow graph from checkpoint.

  Args:
    tf_session: Session to load graph into
    model_folder: Folder containing serialised model
    cp_name: Name of Tensorflow checkpoint
    scope: Variable scope to use.
    verbose: Whether to print additional debugging information. 是否打印额外的调试信息（默认为 False）
  """
  # Load model proper
  load_path = os.path.join(model_folder, '{0}.ckpt'.format(cp_name))

  print('Loading model from {0}'.format(load_path))

  print_weights_in_checkpoint(model_folder, cp_name)

  initial_vars = set( # 获取当前默认图中所有变量的名称，并将其存储在 initial_vars 集合中。这些变量名称在加载前被记录，以便后续对比。
      [v.name for v in tf.get_default_graph().as_graph_def().node])

  # Saver
  if scope is None:
    saver = tf.train.Saver() # 创建一个不带特定变量范围的 tf.train.Saver() 实例
  else: # 获取在指定范围内的全局变量，并将其传递给 Saver
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope) 
    saver = tf.train.Saver(var_list=var_list, max_to_keep=100000)
  # Load
  saver.restore(tf_session, load_path) # 加载指定路径的模型到当前 TensorFlow 会话
  all_vars = set([v.name for v in tf.get_default_graph().as_graph_def().node]) # 加载后，获取当前图中所有变量的名称，并将其存储在 all_vars 集合中

  if verbose:
    print('Restored {0}'.format(','.join(initial_vars.difference(all_vars)))) # Restored：加载后在当前图中缺失的变量。
    print('Existing {0}'.format(','.join(all_vars.difference(initial_vars)))) # Existing：在当前图中存在但未在初始变量中记录的变量。
    print('All {0}'.format(','.join(all_vars))) # All：当前图中的所有变量。
    
  print('Done.')


def print_weights_in_checkpoint(model_folder, cp_name):
  """Prints all weights in Tensorflow checkpoint.

  Args:
    model_folder: Folder containing checkpoint
    cp_name: Name of checkpoint

  Returns:

  """
  load_path = os.path.join(model_folder, '{0}.ckpt'.format(cp_name))

  print_tensors_in_checkpoint_file(
      file_name=load_path,
      tensor_name='',
      all_tensors=True,
      all_tensor_names=True)
