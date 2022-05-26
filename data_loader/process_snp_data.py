from typing import Text, Dict, List

import pandas as pd
import numpy as np

from common import keys


def make_iupac_mapping() -> Dict[Text, np.array]:
  nb_classes = 5
  mapping = {".": 0, "A": 1, "T": 2, "C": 3, "G": 4}

  template_np_array_mapping = {}
  uipac_np_array_mapping = {}
  uipac_code_list = [
      "A", "C", "G", "T", "U", "R", "Y", "S", "W", "K", "M", "B", "D", "H", "V",
      "N", ".", "-"
  ]

  for class_label in mapping:
    template_np_array_mapping[class_label] = np.eye(nb_classes)[
        mapping[class_label]]

  for code in uipac_code_list:
    if code in ["A", "T", "C", "G"]:
      uipac_np_array_mapping[code] = template_np_array_mapping[code]
    elif code == "U":
      uipac_np_array_mapping[code] = template_np_array_mapping["T"]
    elif code == "R":
      uipac_np_array_mapping[code] = (template_np_array_mapping["A"] +
                                      template_np_array_mapping["G"]) / 2
    elif code == "Y":
      uipac_np_array_mapping[code] = (template_np_array_mapping["C"] +
                                      template_np_array_mapping["T"]) / 2
    elif code == "S":
      uipac_np_array_mapping[code] = (template_np_array_mapping["G"] +
                                      template_np_array_mapping["C"]) / 2
    elif code == "W":
      uipac_np_array_mapping[code] = (template_np_array_mapping["A"] +
                                      template_np_array_mapping["T"]) / 2
    elif code == "K":
      uipac_np_array_mapping[code] = (template_np_array_mapping["G"] +
                                      template_np_array_mapping["T"]) / 2
    elif code == "M":
      uipac_np_array_mapping[code] = (template_np_array_mapping["A"] +
                                      template_np_array_mapping["C"]) / 2
    elif code == "B":
      uipac_np_array_mapping[code] = (template_np_array_mapping["C"] +
                                      template_np_array_mapping["G"] +
                                      template_np_array_mapping["T"]) / 3
    elif code == "D":
      uipac_np_array_mapping[code] = (template_np_array_mapping["A"] +
                                      template_np_array_mapping["G"] +
                                      template_np_array_mapping["T"]) / 3
    elif code == "H":
      uipac_np_array_mapping[code] = (template_np_array_mapping["A"] +
                                      template_np_array_mapping["C"] +
                                      template_np_array_mapping["T"]) / 3
    elif code == "V":
      uipac_np_array_mapping[code] = (template_np_array_mapping["A"] +
                                      template_np_array_mapping["C"] +
                                      template_np_array_mapping["G"]) / 3
    elif code == "N":
      uipac_np_array_mapping[code] = (
          template_np_array_mapping["A"] + template_np_array_mapping["C"] +
          template_np_array_mapping["G"] + template_np_array_mapping["T"]) / 4
    else:
      uipac_np_array_mapping[code] = template_np_array_mapping["."]
  return uipac_np_array_mapping


def process_snp_data(data: np.array) -> np.array:
  """process snp data to one-hot code.

  Args:
      data (np.array): input data.

  Returns:
      np.array: one-hot codeed data.
  """
  nb_classes = 4
  onehot_x = np.empty(shape=(data.shape[0], data.shape[1], nb_classes))
  for i in range(0, data.shape[0]):
    _data = pd.to_numeric(data[i], downcast='signed')
    _targets = np.array(_data).reshape(-1)
    onehot_x[i] = np.eye(nb_classes)[_targets]
  return onehot_x


def process_snp_letter_data(data: np.array) -> np.array:
  """process snp data to one-hot code.

  Args:
      data (np.array): input data.

  Returns:
      np.array: one-hot codeed data.
  """
  nb_classes = 5
  mapping = make_iupac_mapping()

  onehot_x = np.empty(shape=(data.shape[0], data.shape[1], nb_classes))
  for i in range(0, data.shape[0]):
    _data = np.array(data[i]).reshape(-1)
    np_value = [mapping[val] for val in _data]
    onehot_x[i] = np_value
  return onehot_x


def load_snp_data_with_multi_labels(
    filename: Text,
    lables: List,
    drop_cols: List = None,
    data_type: Text = "snp") -> Dict[Text, np.array]:
  """load snp data.

  Args:
      filename (Text): filename of dataset.

  Returns:
      pd.DataFrame: dataset.
  """
  dataset = pd.read_csv(filename, na_filter=False, low_memory=False)
  if drop_cols is not None:
    dataset = dataset.drop(columns=drop_cols)

  y = dataset[lables].to_numpy()
  x = dataset.drop(columns=lables).to_numpy()

  if data_type == 'zygosity':
    onehot_x = process_snp_data(x)
  elif data_type == 'snp':
    onehot_x = process_snp_letter_data(x)

  return {keys.KEY_FEATURES: onehot_x, keys.KEY_LABEL: y}