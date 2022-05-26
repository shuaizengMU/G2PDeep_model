"""Metric for TensorFlow/Keras callback function."""
import tensorflow as tf


def pearsonr(x, y, axis=-2):
  """Calculates a Pearson correlation coefficient of two tensors."""
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
