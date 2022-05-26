"""model of dual cnn."""
from typing import Dict, Text

from tensorflow import keras
import numpy as np
import attr


@attr.s(auto_attribs=True)
class ModelHParamsDuallCNN:
  """Model hyperparameters for Dual CNN model.

  Attributes:
  left_tower_filters_list: list of int number. Each element represents number
    of filters. Lenght of it is number of layers. Default to [10, 10].
  left_tower_kernel_size_list: list of int number. Each element represents size
    of kernel for 1D CNN. Lenght of it is number of layers. Default to [4, 20].
  right_tower_filters_list: list of int number. Each element represents number
    of filters. Lenght of it is number of layers. Default to[10].
  right_tower_kernel_size_list: list of int number. Each element represents size
    of kernel for 1D CNN. Lenght of it is number of layers. Default to [4].
  central_tower_filters_list: list of int number. Each element represents number
    of filters. Lenght of it is number of layers. Default to[10].
  central_tower_kernel_size_list: list of int number. Each element represents
    size of kernel for 1D CNN. Lenght of it is number of layers. Default to [4].
  dnn_size_list: list of int number. Each element represents number
    of logits in DNN layer. enght of it is number of layers. Default to [1].
  activation: str, Name of activation. Default to `linear`.
  dropout_rate: float, dropout rate. Default to 0.75.
  """
  left_tower_filters_list: attr.Factory(list) = [10, 10]
  left_tower_kernel_size_list: attr.Factory(list) = [4, 20]
  right_tower_filters_list: attr.Factory(list) = [10]
  right_tower_kernel_size_list: attr.Factory(list) = [4]
  central_tower_filters_list: attr.Factory(list) = [10]
  central_tower_kernel_size_list: attr.Factory(list) = [4]
  dnn_size_list: attr.Factory(list) = [1]
  activation: Text = 'linear'
  dropout_rate: float = 0.75

  @classmethod
  def from_dict(cls, hparams: Dict):
    return cls(
        left_tower_filters_list=hparams['left_tower_filters_list'],
        left_tower_kernel_size_list=hparams[
            'left_tower_kernel_size_list'],
        right_tower_filters_list=hparams['right_tower_filters_list'],
        right_tower_kernel_size_list=hparams[
            'right_tower_kernel_size_list'],
        central_tower_filters_list=hparams[
            'central_tower_filters_list'],
        central_tower_kernel_size_list=hparams[
            'central_tower_kernel_size_list'],
        dnn_size_list=hparams['dnn_size_list'],
    )


def make_dual_cnn_model(input_data: np.array, num_feature_dim: int,
                        model_hyperparams: ModelHParamsDuallCNN) -> keras.Model:
  """make dual-cnn model.

  The model is CNN based model published in Yang's paper [1].

  citation:
    [1] Liu, Yang, Duolin Wang, Fei He, Juexin Wang, Trupti Joshi, and Dong Xu.
        "Phenotype prediction and genome-wide association study using deep
        convolutional neural network of soybean."
        Frontiers in genetics (2019): 1091.

  Args:
      input_data (np.array): Input data.
      num_feature_dim (int): number of feature dimension.
      model_hyperparams (ModelHParamsDuallCNN): hyperparameters for model.

  Returns:
      keras.Model: dual CNN model.
  """

  inputs = keras.Input(shape=(input_data.shape[1], num_feature_dim),
                       name='input')

  left_tower_x = inputs
  right_tower_x = inputs

  # left tower
  num_layers = len(model_hyperparams.left_tower_filters_list)
  for idx in range(num_layers):
    left_tower_x = keras.layers.Conv1D(
        filters=model_hyperparams.left_tower_filters_list[idx],
        kernel_size=model_hyperparams.left_tower_kernel_size_list[idx],
        padding='same',
        activation=model_hyperparams.activation,
        kernel_initializer='TruncatedNormal',
        kernel_regularizer=keras.regularizers.l2(0.1),
        bias_regularizer=keras.regularizers.l2(0.01),
        name=f'left_tower_{idx}')(left_tower_x)

  # right tower
  num_layers = len(model_hyperparams.right_tower_filters_list)
  for idx in range(num_layers):
    right_tower_x = keras.layers.Conv1D(
        filters=model_hyperparams.right_tower_filters_list[idx],
        kernel_size=model_hyperparams.right_tower_kernel_size_list[idx],
        padding='same',
        activation=model_hyperparams.activation,
        kernel_initializer='TruncatedNormal',
        kernel_regularizer=keras.regularizers.l2(0.1),
        bias_regularizer=keras.regularizers.l2(0.01),
        name=f'right_tower_{idx}')(right_tower_x)

  x = keras.layers.add([left_tower_x, right_tower_x])

  # central tower
  num_layers = len(model_hyperparams.central_tower_filters_list)
  for idx in range(num_layers):
    x = keras.layers.Conv1D(
        filters=model_hyperparams.central_tower_filters_list[idx],
        kernel_size=model_hyperparams.central_tower_kernel_size_list[idx],
        padding='same',
        activation=model_hyperparams.activation,
        kernel_initializer='TruncatedNormal',
        kernel_regularizer=keras.regularizers.l2(0.1),
        bias_regularizer=keras.regularizers.l2(0.2),
        name=f'central_tower_{idx}')(x)

  x = keras.layers.Dropout(model_hyperparams.dropout_rate)(x)
  x = keras.layers.Flatten()(x)
  x = keras.layers.Dropout(model_hyperparams.dropout_rate)(x)

  num_layers = len(model_hyperparams.dnn_size_list) - 1
  for idx in range(num_layers):
    x = keras.layers.Dense(model_hyperparams.dnn_size_list[idx],
                           activation=model_hyperparams.activation,
                           bias_regularizer=keras.regularizers.l2(0.2),
                           kernel_initializer='TruncatedNormal',
                           name=f'dnn_{idx}')(x)

  outputs = keras.layers.Dense(model_hyperparams.dnn_size_list[-1],
                               activation=model_hyperparams.activation,
                               bias_regularizer=keras.regularizers.l2(0.2),
                               kernel_initializer='TruncatedNormal',
                               name='output')(x)

  model = keras.Model(inputs=inputs, outputs=outputs)
  return model
