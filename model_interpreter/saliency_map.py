from third_party.vis.utils import utils
import numpy as np

from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt

import calendar
import time
import os

tf.compat.v1.disable_eager_execution()


def compile_saliency_function(model):
  layer_index = utils.find_layer_idx(model, 'output')
  inp = model.layers[0].input
  outp = model.layers[layer_index].output

  max_outp = keras.backend.max(outp, axis=1)
  # sum_res = keras.backend.sum(max_outp)
  saliency = keras.backend.gradients(keras.backend.sum(max_outp), inp)
  return keras.backend.function([inp, keras.backend.learning_phase()], saliency)


def get_saliency(testSNP, model):

  array = np.array([testSNP])
  saliency_fn = compile_saliency_function(model)
  saliency_out = saliency_fn([[y for y in array][0], 1])
  saliency = saliency_out[0]
  saliency = saliency[::-1].transpose(1, 0, 2)
  output = np.abs(saliency).max(axis=-1)

  return output


def save_images_plot(saliency, filename):
  plt.figure(figsize=(15, 8), facecolor='w')
  plt.plot(saliency, 'b.')

  line = sorted(saliency, reverse=True)[10]
  plt.axhline(y=line, color='b', linestyle='--')
  plt.ylabel('saliency value', fontdict=None, labelpad=None, fontsize=15)
  plt.savefig(filename)
  plt.clf()
  plt.cla()
  plt.close()


def run_saliency_map(model: keras.Model, data: np.array,
                     filename_path: str) -> np.array:
  """[summary]

  Args:
      model (keras.Model): [description]
      data (np.array): [description]

  Returns:
      np.array: [description]
  """
  saliency = get_saliency(data, model)

  saliency = np.median(saliency, axis=-1)

  save_images_plot(saliency, filename_path)

  # samples_idx = sorted(range(len(saliency)), key=lambda i: saliency[i])[-20:]
  saliency_list = saliency.tolist()
  samples_idx = range(len(saliency_list))

  data = []
  for idx in samples_idx:
    data.append([idx, saliency_list[idx]])

  return data


def run_raw_saliency_map(model: keras.Model, data: np.array) -> np.array:
  """[summary]

  Args:
      model (keras.Model): [description]
      data (np.array): [description]

  Returns:
      np.array: [description]
  """
  return get_saliency(data, model)