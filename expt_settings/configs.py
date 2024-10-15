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

"""Default configs for TFT experiments.

Contains the default output paths for data, serialised models and predictions
for the main experiments used in the publication.
"""

import os

import data_formatters.electricity
import data_formatters.favorita
import data_formatters.traffic
import data_formatters.volatility


class ExperimentConfig(object):
  """Defines experiment configs and paths to outputs.

  Attributes:
    root_folder: Root folder to contain all experimental outputs.
    experiment: Name of experiment to run.
    data_folder: Folder to store data for experiment.
    model_folder: Folder to store serialised models.
    results_folder: Folder to store results.
    data_csv_path: Path to primary data csv file used in experiment.
    hyperparam_iterations: Default number of random search iterations for
      experiment.
  """

  default_experiments = ['volatility', 'electricity', 'traffic', 'favorita']

  def __init__(self, experiment='volatility', root_folder=None):
    """Creates configs based on default experiment chosen.

    Args:
      experiment: Name of experiment.
      root_folder: Root folder to save all outputs of training.
    """

    if experiment not in self.default_experiments:
      raise ValueError('Unrecognised experiment={}'.format(experiment))

    # Defines all relevant paths
    if root_folder is None:
      root_folder = os.path.join(
          os.path.dirname(os.path.realpath(__file__)), '..', 'outputs')
      print('Using root folder {}'.format(root_folder))
    """
    os.path.realpath(__file__)：获取当前 Python 文件的绝对路径。__file__ 是 Python 中的一个特殊变量，表示当前文件的路径。
    os.path.dirname()：获取该文件的所在目录。通过 os.path.realpath(__file__) 获取当前文件的完整路径后，os.path.dirname() 会返回该路径的目录部分。
    '..'：表示上级目录。代码中的 os.path.join(..., '..') 将当前文件的路径向上一级。
    'outputs'：定义了存储输出文件的文件夹名称。
    os.path.join(...)：将所有路径片段连接在一起，最终结果是将根目录设置为当前 Python 文件的上一级目录中的一个 outputs 文件夹。
    print(): 输出生成的默认路径，帮助用户了解程序自动选择的 root_folder。
    """

    # eg：experiment=electricity
    # 生成outputs/data/electricity；outputs/saved_models/electricity；outputs/results/electricity
    self.root_folder = root_folder
    self.experiment = experiment
    self.data_folder = os.path.join(root_folder, 'data', experiment)
    self.model_folder = os.path.join(root_folder, 'saved_models', experiment)
    self.results_folder = os.path.join(root_folder, 'results', experiment)

    # Creates folders if they don't exist
    for relevant_directory in [
        self.root_folder, self.data_folder, self.model_folder,
        self.results_folder
    ]:
      if not os.path.exists(relevant_directory):
        os.makedirs(relevant_directory)
 
  @property
  def data_csv_path(self):
    csv_map = {
        'volatility': 'formatted_omi_vol.csv',
        'electricity': 'hourly_electricity.csv',
        'traffic': 'hourly_data.csv',
        'favorita': 'favorita_consolidated.csv'
    }

    return os.path.join(self.data_folder, csv_map[self.experiment])

  @property
  def hyperparam_iterations(self):
    return 240 if self.experiment == 'volatility' else 60
  """
  volatility默认进行 240 次随机搜索迭代来优化超参数。 对于其他实验，默认只需要 60 次超参数搜索迭代。
  @property 是一个 Python 内置的装饰器，作用是将类的方法变成一个属性，从而可以像访问属性一样调用它，而不需要加括号 () 
  eg:
  config = ExperimentConfig(experiment='volatility')
  print(config.hyperparam_iterations)  # 输出：240
  eg:
  config = ExperimentConfig(experiment='electricity')
  print(config.hyperparam_iterations)  # 输出：60
  """

  def make_data_formatter(self):
    """Gets a data formatter object for experiment. 数据格式化器用于为每个实验准备和处理数据，使其适合后续的模型训练或评估。

    Returns:
      Default DataFormatter per experiment.
    """

    data_formatter_class = {
        'volatility': data_formatters.volatility.VolatilityFormatter,
        'electricity': data_formatters.electricity.ElectricityFormatter,
        'traffic': data_formatters.traffic.TrafficFormatter,
        'favorita': data_formatters.favorita.FavoritaFormatter
    }

    return data_formatter_class[self.experiment]()
  """
    config = ExperimentConfig(experiment='electricity')
    formatter = config.make_data_formatter()
    print(type(formatter))  # 输出：<class 'data_formatters.electricity.ElectricityFormatter'>
  """
