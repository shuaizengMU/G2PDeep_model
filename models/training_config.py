"""Configure of experiment."""
from typing import Dict, Text

import attr


@attr.s(auto_attribs=True)
class TrainingHParams:
  """
  Arrtributes:
    learning_rate: int, learning rate. Defaul is 0.001.
    epochs: int, epochs for model training. Default is 100.
    optimizer: str, name of optimizer for model. Default is `adam`.
    loss: str, name of loss function. Default is `mse`.
    metrics: str, name of metrics to evaluate model. Default is `mae`.
    batch_size: int, batch size. Default is 32.
    early_stopping_patience: int, patience in early stopping. Default is 5
  """
  learning_rate: float = 0.0001
  epochs: int = 1
  optimizer: Text = 'adam'
  loss: Text = 'mse'
  metrics: Text = 'mae'
  batch_size: int = 32
  early_stopping_patience = 5

  @classmethod
  def from_dict(cls, training_hparams: Dict):
    """extracting values from model.
    Args:
        query_set (Dict): query set from model.
    Returns:
        db_config: initalized training hyperparameters.
    """
    return cls(
        learning_rate=training_hparams['learning_rate'],
        epochs=training_hparams['epochs'],
        optimizer=training_hparams['optimizer'],
        loss=training_hparams['loss'],
        metrics=training_hparams['metrics'],
        batch_size=training_hparams['batch_size'],
        early_stopping_patience=training_hparams['early_stopping_patience'],
    )
