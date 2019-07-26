# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

"""Data generators for En-Vi translation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

_ENRU_TRAIN_DATASETS = [[
    "gs://t2t_translate_enru/train.zip",
    ("train-en.txt","train-ru.txt")
]]

_ENRU_TEST_DATASETS = [[
    "gs://t2t_translate_enru/test.zip",
    ("test-en.txt","test-ru.txt")
]]


# See this PR on github for some results with Transformer on this Problem.
# https://github.com/tensorflow/tensor2tensor/pull/611


@registry.register_problem
class TranslateEnruIwslt2k(translate.TranslateProblem):

  @property
  def approx_vocab_size(self):
    return 2**11

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENRU_TRAIN_DATASETS if train else _ENRU_TEST_DATASETS

@registry.register_problem
class TranslateEnruIwslt32k(translate.TranslateProblem):

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENRU_TRAIN_DATASETS if train else _ENRU_TEST_DATASETS


@registry.register_problem
class TranslateEnruIwslt8k(translate.TranslateProblem):

  @property
  def approx_vocab_size(self):
    return 2**13

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENRU_TRAIN_DATASETS if train else _ENRU_TEST_DATASETS

@registry.register_problem
class TranslateEnruIwslt16k(translate.TranslateProblem):

  @property
  def approx_vocab_size(self):
    return 2**14  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENRU_TRAIN_DATASETS if train else _ENRU_TEST_DATASETS

@registry.register_problem
class TranslateEnruIwslt128k(translate.TranslateProblem):

  @property
  def approx_vocab_size(self):
    return 2**17  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENRU_TRAIN_DATASETS if train else _ENRU_TEST_DATASETS
