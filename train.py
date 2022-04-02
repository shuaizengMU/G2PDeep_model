from typing import Dict, Text

from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import attr

import evaluation_util
import load_dataset_util
import saliency_map

import tensorflow as tf
import os

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './data/SoyNAM', 'Dirctory of dataset')
flags.DEFINE_string('result_dir', './results', 'Dirctory of output')
flags.DEFINE_string('dataset_type', 'height',
                    'Type of dataset (height|oil|moisture|protein|yield).')


# Model hyperparameters
@attr.s(auto_attribs=True)
class model_hyperparameters:
  """
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
  left_tower_filters_list: attr.Factory(list) = [10, 10]  # type: ignore
  left_tower_kernel_size_list: attr.Factory(list) = [4, 20]  # type: ignore
  right_tower_filters_list: attr.Factory(list) = [10]  # type: ignore
  right_tower_kernel_size_list: attr.Factory(list) = [4]  # type: ignore
  central_tower_filters_list: attr.Factory(list) = [10]  # type: ignore
  central_tower_kernel_size_list: attr.Factory(list) = [4]  # type: ignore
  dnn_size_list: attr.Factory(list) = [1]  # type: ignore
  activation: Text = "linear"
  dropout_rate: float = 0.75

  @classmethod
  def from_dict(cls, model_hyperparams: Dict):
    return cls(
        left_tower_filters_list=model_hyperparams['left_tower_filters_list'],
        left_tower_kernel_size_list=model_hyperparams[
            'left_tower_kernel_size_list'],
        right_tower_filters_list=model_hyperparams['right_tower_filters_list'],
        right_tower_kernel_size_list=model_hyperparams[
            'right_tower_kernel_size_list'],
        central_tower_filters_list=model_hyperparams[
            'central_tower_filters_list'],
        central_tower_kernel_size_list=model_hyperparams[
            'central_tower_kernel_size_list'],
        dnn_size_list=model_hyperparams['dnn_size_list'],
    )  # type: ignore


@attr.s(auto_attribs=True)
class experimental_parameters:
  """
  Arrtributes:
    learning_rate: int, learning rate. Defaul is 0.001.
    epochs: int, epochs for model training. Default is 100.
    optimizer: str, name of optimizer for model. Default is `adam`.
    loss: str, name of loss function. Default is `mse`.
    metrics: str, name of metrics to evaluate model. Default is `mae`.
    batch_size: int, batch size. Default is 32.
  """
  learning_rate: float = 0.0001
  epochs: int = 250
  optimizer: Text = 'adam'
  loss: Text = 'mse'
  metrics: Text = 'mae'
  batch_size: int = 32

  @classmethod
  def from_dict(cls, experimental_parameters: Dict):
    """extracting values from model.

    Args:
        query_set (Dict): query set from model.

    Returns:
        db_config: initalized experimental_parameters.
    """
    return cls(
        learning_rate=experimental_parameters['learning_rate'],
        epochs=experimental_parameters['epochs'],
        optimizer=experimental_parameters['optimizer'],
        loss=experimental_parameters['loss'],
        metrics=experimental_parameters['metrics'],
        batch_size=experimental_parameters['batch_size'],
    )


def correlationMetric(x, y, axis=-2):
  """Metric returning the PPC of two tensors over some axis, default -2."""
  x = tf.convert_to_tensor(x)
  y = tf.cast(y, x.dtype)
  n = tf.cast(tf.shape(x)[axis], x.dtype)
  xsum = tf.reduce_sum(x, axis=axis)
  ysum = tf.reduce_sum(y, axis=axis)
  xmean = xsum / n
  ymean = ysum / n
  xvar = tf.reduce_sum(tf.math.squared_difference(x, xmean), axis=axis)
  yvar = tf.reduce_sum(tf.math.squared_difference(y, ymean), axis=axis)
  cov = tf.reduce_sum((x - xmean) * (y - ymean), axis=axis)
  corr = cov / tf.sqrt(xvar * yvar)
  return tf.constant(1.0, dtype=x.dtype) - corr


def get_deep_gwas_model(
    input_data: np.array,
    model_hyperparams: model_hyperparameters) -> keras.Model:

  num_classes = 4
  inputs = keras.Input(shape=(input_data.shape[1], num_classes), name="input")

  left_tower_x = inputs
  right_tower_x = inputs

  # left tower
  _num_layers = len(model_hyperparams.left_tower_filters_list)
  for idx in range(_num_layers):
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
  _num_layers = len(model_hyperparams.right_tower_filters_list)
  for idx in range(_num_layers):
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
  _num_layers = len(model_hyperparams.central_tower_filters_list)
  for idx in range(_num_layers):
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

  _num_layers = len(model_hyperparams.dnn_size_list) - 1
  for idx in range(_num_layers):
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


def train_model() -> keras.Model:
  """train model
    Returns:
      keras.Model: trained keras model.
  """

  training_filename = os.path.join(FLAGS.data_dir,
                                   "%s.train.csv" % (FLAGS.dataset_type))
  test_filename = os.path.join(FLAGS.data_dir,
                               "%s.test.csv" % (FLAGS.dataset_type))

  dataset_dict = load_dataset_util.load_snp_data_with_multi_labels(
      training_filename, ["label"], data_type='zygosity')

  x_other = dataset_dict[load_dataset_util.KEY_FEATURES]
  y_other = dataset_dict[load_dataset_util.KEY_LABEL]

  dataset_dict = load_dataset_util.load_snp_data_with_multi_labels(
      test_filename, ["label"], data_type='zygosity')

  x_test = dataset_dict[load_dataset_util.KEY_FEATURES]
  y_test = dataset_dict[load_dataset_util.KEY_LABEL]

  # y_other = stats.zscore(y_other)
  x_train, x_valid, y_train, y_valid = train_test_split(x_other,
                                                        y_other,
                                                        test_size=len(x_test))

  # define model
  experimental_params = experimental_parameters()
  model_hyperparams = model_hyperparameters()

  # model = model_deep_gwas.DeepGwasModel(model_hyperparams)
  model = get_deep_gwas_model(x_train, model_hyperparams)
  model.compile(loss=experimental_params.loss,
                optimizer=experimental_params.optimizer,
                metrics=[experimental_params.metrics, correlationMetric])

  early_stopping = EarlyStopping(monitor='val_mae',
                                 patience=6,
                                 restore_best_weights=True)
  model.summary()

  # model training
  model.fit(x=x_train,
            y=y_train,
            batch_size=experimental_params.batch_size,
            epochs=experimental_params.epochs,
            validation_data=(x_valid, y_valid),
            verbose=1,
            callbacks=[early_stopping])

  # get predicted values for traning and validation dataset.
  pred_train_y = model.predict(x_train)
  pred_valid_y = model.predict(x_valid)
  pred_test_y = model.predict(x_test)

  # generate performance
  output_filename = os.path.join(FLAGS.result_dir,
                                 f"performance_{FLAGS.dataset_type}.csv")
  evaluation_util.save_performance(pred_train_y, y_train, pred_valid_y, y_valid,
                                   pred_test_y, y_test, output_filename)

  # generate saliency map
  output_filename = os.path.join(FLAGS.result_dir,
                                 f"saliency_{FLAGS.dataset_type}.png")
  saliency_data = saliency_map.run_saliency_map(model, x_test,
                                                output_filename)
  saliency_df = pd.DataFrame.from_records(saliency_data,
                                          columns=["index", "saliency"])
  
  output_filename = os.path.join(FLAGS.result_dir,
                                 f"saliency_{FLAGS.dataset_type}_test.csv")
  saliency_df.to_csv(output_filename, index=False)


def main(argv):

  tf.random.set_seed(1)
  np.random.seed(1)

  train_model()


if __name__ == "__main__":
  app.run(main)
