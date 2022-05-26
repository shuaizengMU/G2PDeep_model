"""utility for calculate the metrics for regression and classification."""
from typing import Dict, Text

from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from common import keys

import numpy as np
import pandas as pd


def calc_regression_metrics(values_pred: np.array,
                            values_true: np.array) -> Dict[Text, float]:
  """Calculate PCC, MSE and MAE.

  Clculating the Pearson Corelation Coefficient, Mean Squared Error and Mean
  Absolute Error for given two arrays.

  Args:
      values_pred (np.array): first values from array.
      values_true (np.array): second values from array.

  Returns:
      Dict: A dictionary of string to corresponding values.
  """
  data = {}
  data[keys.KEY_PCC] = stats.spearmanr(values_pred, values_true)[0]
  data[keys.KEY_MSE] = mean_squared_error(values_pred, values_true)
  data[keys.KEY_MAE] = mean_absolute_error(values_pred, values_true)
  data[keys.KEY_R_SQUARE] = r2_score(values_true, values_pred)

  return data


def save_performance(pred_train_y: np.array, y_train: np.array,
                     pred_valid_y: np.array, y_valid: np.array,
                     pred_test_y: np.array, y_test: np.array,
                     output_filename: str):
  """Calculatig and saving performance resaults.

  Args:
      pred_train_y (np.array): prediction results for training dataset.
      y_train (np.array): true labels for training dataset.
      pred_valid_y (np.array): prediction results for validation dataset.
      y_valid (np.array): true labels for validation dataset.
      pred_test_y (np.array): prediction results for test dataset.
      y_test (np.array): true labels for test dataset.
  """

  metrics_train_dict = calc_regression_metrics(pred_train_y, y_train)
  metrics_validation_dict = calc_regression_metrics(pred_valid_y, y_valid)
  metrics_test_dict = calc_regression_metrics(pred_test_y, y_test)

  result_list = []
  result_list.append(pd.DataFrame(metrics_train_dict, index=[0]))
  result_list.append(pd.DataFrame(metrics_validation_dict, index=[1]))
  result_list.append(pd.DataFrame(metrics_test_dict, index=[2]))

  result_df = pd.concat(result_list)
  result_df['dataset'] = ['training', 'validation', 'test']
  result_df.to_csv(output_filename, index=False)
  print(result_df)