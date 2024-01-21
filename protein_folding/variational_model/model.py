import collections

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

tfd = tfp.distributions

def _SaveModel(model, location):
  model.save(location, overwrite=True, save_format='tf',
      options=tf.saved_model.SaveOptions())

def _SaveWeights(model, location):
  model.save_weights(location, overwrite=True, save_format='tf',
      options=tf.train.CheckpointOptions())

def _XMask(x):
    return tf.cast(
            tf.math.reduce_any(
                tf.math.greater(tf.math.abs(x), 1e-6), axis=[-1]), tf.float32)

class Conditioner(object):
  def __init__(self, model):
    self._model = model

  def conditioning(self, residue_names, atom_names, training):
    return self._model({
      'residue_names': residue_names,
      'atom_names': atom_names}, training=training)

  def trainable_weights(self):
    return self._model.trainable_weights

  def save(self, location):
    _SaveModel(self._model, location)

  def save_weights(self, location):
    _SaveWeights(self._model, location)

class Decoder(object):
  def  __init__(self, model):
    self._model = model

  def decode(self, z, atom_mask, cond, training):
    outputs = self._model({
      'z': z,
      'atom_mask': atom_mask,
      'cond': cond}, training=training)
    return tfd.MultivariateNormalDiag(
        loc=outputs[0], scale_diag=outputs[1])

  def trainable_weights(self):
    return self._model.trainable_weights

  def save(self, location):
    _SaveModel(self._model, location)

  def save_weights(self, location):
    _SaveWeights(self._model, location)

class Encoder(object):
  def __init__(self, model):
    self._model = model

  # Encodes a latent representation of the structure.
  #
  # Args:
  #  normalized_coordinates: The atom's coordinates, normalized to have mean 0.
  #    Should have dimensions: (batch_size, num_proteins, num_atoms, 3)
  #  cond: A tensor with context information. Should have dimensions:
  #    (batch_size, num_proteins, num_atoms, num_channels)
  #
  # Returns a latent representation of the structure.
  def encode(self, normalized_coordinates, atom_mask, cond, training):
    return tf.split(
        self._model({
          'normalized_coordinates': normalized_coordinates,
          'atom_mask': atom_mask,
          'cond': cond}, training=training), num_or_size_splits=2, axis=1)

  # Reparametrizes mean and logvar to z.
  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * 0.5) + mean

  def trainable_weights(self):
    return self._model.trainable_weights

  def save(self, location):
    _SaveModel(self._model, location)

  def save_weights(self, location):
    _SaveWeights(self._model, location)

LossInformation = collections.namedtuple(
    'LossInformation', ['loss', 'logpx_z', 'logpz', 'logqz_x', 'diff_mae'])
class VariationalModel(object):
  def __init__(self, conditioner, decoder, encoder):
    self._conditioner = conditioner
    self._decoder = decoder
    self._encoder = encoder

  def _log_normal_pdf(self, sample, mean, logvar, raxis=[1]):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -0.5*((sample-mean)**2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

  def trainable_weights(self):
    return (
        self._conditioner.trainable_weights() +
        self._decoder.trainable_weights() +
        self._encoder.trainable_weights())

  def compute_loss(self, training_data, training):
    atom_mask = _XMask(training_data['normalized_coordinates'])
    cond = self._conditioner.conditioning(
        training_data['residue_names'], training_data['atom_names'], training)
    mean, logvar = self._encoder.encode(
        training_data['normalized_coordinates'], atom_mask, cond, training)
    z = self._encoder.reparameterize(mean, logvar)
    x = self._decoder.decode(z, atom_mask, cond, training)
    # See https://www.tensorflow.org/tutorials/generative/cvae#define_the_loss_function_and_the_optimizer
    # for definition of the loss functions.
    logpx_z = tf.reduce_sum(
            tf.math.multiply(
                x.log_prob(training_data['normalized_coordinates']),
                atom_mask), axis=[-1, -2])
    logpz = self._log_normal_pdf(z, 0., 0.)
    logqz_x = self._log_normal_pdf(z, mean, logvar)
    diff_mae = tf.math.reduce_sum(
            tf.math.multiply(
                tf.math.abs(training_data['normalized_coordinates'] - x.mean()),
                tf.expand_dims(atom_mask, -1)))/tf.math.reduce_sum(atom_mask)

    return LossInformation(
        loss=-tf.reduce_mean(logpx_z + logpz - logqz_x),
        logpx_z=tf.reduce_mean(logpx_z),
        logpz=tf.reduce_mean(logpz),
        logqz_x=tf.reduce_mean(logqz_x),
        diff_mae=diff_mae)

  def save(self, location):
    self._conditioner.save(location + '/conditioner')
    self._decoder.save(location + '/decoder')
    self._encoder.save(location + '/encoder')
  
  def save_weights(self, location):
    self._conditioner.save_weights(location + '/conditioner')
    self._decoder.save_weights(location + '/decoder')
    self._encoder.save_weights(location + '/encoder')
