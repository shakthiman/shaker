import collections

import tensorflow as tf
import tf_keras

import numpy as np

from tensorflow_graphics.geometry.transformation import rotation_matrix_3d

def _SaveModel(model, location):
  model.save(location + '.keras', overwrite=True)

def _SaveWeights(model, location):
  model.save_weights(location + '.weights.h5', overwrite=True)

def _XMask(x):
    return tf.cast(
            tf.math.reduce_any(
                tf.math.greater(tf.math.abs(x), 1e-6), axis=[-1]), tf.float32)

def LoadModel(full_model_location, model_weight_location, suffix):
  model = tf_keras.models.load_model(full_model_location + suffix)
  model.load_weights(model_weight_location + suffix)
  return model

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

  def load_weights(self, location):
    self._model.load_weights(location)

class Distribution(object):
  def __init__(self, loc, scale_diag):
    self._loc = loc
    self._scale_diag = scale_diag

  def mean(self):
    return self._loc

  def log_prob(self, x):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -0.5 * (((self._loc - x) / self._scale_diag)**2.)
        - 0.5 * log2pi
        - tf.math.log(self._scale_diag), axis=-1)

class Decoder(object):
  def  __init__(self, model):
    self._model = model

  def decode(self, z, atom_mask, cond, training):
    z_dict = {'z_' + str(i): z[i]
              for i in range(len(z))}
    outputs = self._model(z_dict | {
      'atom_mask': atom_mask,
      'cond': cond}, training=training)
    return Distribution(
        loc=outputs[0], scale_diag=outputs[1])

  def trainable_weights(self):
    return self._model.trainable_weights

  def save(self, location):
    _SaveModel(self._model, location)

  def save_weights(self, location):
    _SaveWeights(self._model, location)

  def load_weights(self, location):
    self._model.load_weights(location)

class Encoder(object):
  def __init__(self, model, num_outputs):
    self._model = model
    self._num_outputs = num_outputs

  def num_outputs(self):
    return self._num_outputs

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
    return self._model({
          'normalized_coordinates': normalized_coordinates,
          'atom_mask': atom_mask,
          'cond': cond}, training=training)

  # Reparametrizes mean and logvar to z.
  def reparameterize(self, means, logvars):
    reparametrized_outputs = []
    for i in range(len(means)):
      mean = means[i]
      logvar = logvars[i]
      eps = tf.random.normal(shape=tf.shape(mean))
      reparametrized_outputs.append(
          eps * tf.exp(logvar * 0.5) + mean)
    return reparametrized_outputs

  def trainable_weights(self):
    return self._model.trainable_weights

  def save(self, location):
    _SaveModel(self._model, location)

  def save_weights(self, location):
    _SaveWeights(self._model, location)

  def load_weights(self, location):
    self._model.load_weights(location)

class LocalTransformationModel(object):
  def __init__(self, local_coordinates_fn, local_mask_fn, global_coordinates_fn):
    self._local_coordinates_fn = local_coordinates_fn
    self._local_mask_fn = local_mask_fn
    self._global_coordinates_fn = global_coordinates_fn

  # Returns the normalized_coordinates rotated as close as possible to the predicted_coordinates.
  def local_transform(self, normalized_coordinates, predicted_coordinates, atom_mask):
    local_normalized_coordinates = self._local_coordinates_fn(normalized_coordinates)
    local_atom_mask = self._local_mask_fn(atom_mask)
    local_predicted_coordinates = self._local_coordinates_fn(predicted_coordinates)

    num_local_atoms = tf.math.reduce_sum(local_atom_mask, axis=-1, keepdims=True)

    local_normalized_coordinates_mean_removed = local_normalized_coordinates - tf.math.divide_no_nan(
        tf.math.reduce_sum(
          local_normalized_coordinates*tf.expand_dims(local_atom_mask, -1), axis=-2,keepdims=True),
        tf.expand_dims(num_local_atoms, -1))

    local_predicted_coordinates_mean = tf.math.divide_no_nan(
        tf.math.reduce_sum(
          local_predicted_coordinates*tf.expand_dims(local_atom_mask, -1), axis=-2, keepdims=True),
        tf.expand_dims(num_local_atoms, -1))

    local_normalized_coordinates = local_normalized_coordinates_mean_removed + local_predicted_coordinates_mean

    return self._global_coordinates_fn(local_normalized_coordinates)

  def trainable_weights(self):
    return []

  def save(self, location):
    pass

  def save_weights(self, location):
    pass

  def load_weights(self, location):
    pass


LossInformation = collections.namedtuple(
    'LossInformation', ['loss', 'loss_beta_1', 'local_logpx_z', 'logpx_z', 'logpz', 'logqz_x', 'diff_mae', 'local_diff_mae'])
class VariationalModel(object):
  def __init__(self, conditioner, decoder, encoder, rotation_model=None, local_transformation_model=None):
    self._conditioner = conditioner
    self._decoder = decoder
    self._encoder = encoder
    self._rotation_model = rotation_model
    self._local_transformation_model = local_transformation_model

  def _log_normal_pdf_scalar(self, samples, mean, logvar):
    log2pi = tf.math.log(2. * np.pi)
    logprob = 0
    for sample in samples:
      logprob = logprob + tf.reduce_sum(
        -0.5*((sample-mean)**2. * tf.exp(-logvar) + logvar + log2pi),
        axis=list(range(1, len(sample.get_shape().as_list()))))
    return logprob

  def _log_normal_pdf(self, samples, means, logvars):
    log2pi = tf.math.log(2. * np.pi)
    logprob = 0
    for i in range(len(samples)):
      sample = samples[i]
      mean = means[i]
      logvar = logvars[i]
      logprob = logprob + tf.reduce_sum(
        -0.5*((sample-mean)**2. * tf.exp(-logvar) + logvar + log2pi),
        axis=list(range(1, len(sample.get_shape().as_list()))))
    return logprob

  def trainable_weights(self):
    return (
        self._conditioner.trainable_weights() +
        self._decoder.trainable_weights() +
        self._encoder.trainable_weights() +
        (self._rotation_model.trainable_weights if self._rotation_model is not None else []) +
        (self._local_transformation_model.trainable_weights()
         if self._local_transformation_model is not None else []))

  def decode(self, encoder_embedding, training_data, training):
    atom_mask = _XMask(training_data['normalized_coordinates'])
    cond = self._conditioner.conditioning(
        training_data['residue_names'], training_data['atom_names'], training)
    return self._decoder.decode(encoder_embedding, atom_mask, cond, training)

  def encode(self, training_data, training):
    atom_mask = _XMask(training_data['normalized_coordinates'])
    cond = self._conditioner.conditioning(
        training_data['residue_names'], training_data['atom_names'], training)
    means, logvars = self._encoder.encode(
        training_data['normalized_coordinates'], atom_mask, cond, training)
    return self._encoder.reparameterize(mean, logvar)

  def _get_rotation_matrix(self, normalized_coordinates,
                           predicted_coordinates):
    atom_mask = _XMask(normalized_coordinates)
    euler_rotation = self._rotation_model({
      'normalized_coordinates': normalized_coordinates,
      'atom_mask':atom_mask,
      'predicted_coordinates':predicted_coordinates
      })
    euler_rotation = tf.ensure_shape(euler_rotation, [None, 3])
    return rotation_matrix_3d.from_euler(euler_rotation)

  def compute_loss(self, training_data, training, beta=1.0):
    atom_mask = _XMask(training_data['normalized_coordinates'])
    cond = self._conditioner.conditioning(
        training_data['residue_names'], training_data['atom_names'], training)
    encoder_outputs = self._encoder.encode(
        training_data['normalized_coordinates'], atom_mask, cond, training)
    means = [encoder_outputs[i] for i in range(self._encoder.num_outputs())]
    logvars = [encoder_outputs[self._encoder.num_outputs() + i]
               for i in range(self._encoder.num_outputs())]
    z = self._encoder.reparameterize(means, logvars)
    x = self._decoder.decode(z, atom_mask, cond, training)
    # See https://www.tensorflow.org/tutorials/generative/cvae#define_the_loss_function_and_the_optimizer
    # for definition of the loss functions.
    normalized_coordinates = training_data['normalized_coordinates']
    local_logpx_z = 0
    local_diff_mae = 0

    if self._rotation_model is not None:
      rot_matrix = self._get_rotation_matrix(
          normalized_coordinates, x.mean())
      normalized_coordinates = tf.matmul(
              tf.expand_dims(tf.expand_dims(rot_matrix, 1), 1),
              tf.expand_dims(normalized_coordinates, -1))
      normalized_coordinates = tf.squeeze(normalized_coordinates, axis=-1)

    if self._local_transformation_model is not None:
      locally_adjusted_normalized_coordinates = (
          self._local_transformation_model.local_transform(
            normalized_coordinates, x.mean(), atom_mask))
      local_logpx_z = tf.reduce_sum(
          tf.math.multiply(
            x.log_prob(locally_adjusted_normalized_coordinates),
            atom_mask), axis=[-1, -2])
      local_diff_mae = tf.math.reduce_sum(
            tf.math.multiply(
                tf.math.abs(locally_adjusted_normalized_coordinates - x.mean()),
                tf.expand_dims(atom_mask, -1)))/tf.math.reduce_sum(atom_mask)

    logpx_z = tf.reduce_sum(
            tf.math.multiply(
                x.log_prob(normalized_coordinates),
                atom_mask), axis=[-1, -2])
    logpz = self._log_normal_pdf_scalar(z, 0., 0.)
    logqz_x = self._log_normal_pdf(z, means, logvars)
    diff_mae = tf.math.reduce_sum(
            tf.math.multiply(
                tf.math.abs(normalized_coordinates - x.mean()),
                tf.expand_dims(atom_mask, -1)))/tf.math.reduce_sum(atom_mask)

    return LossInformation(
        loss=-tf.reduce_mean(logpx_z + local_logpx_z + beta*(logpz - logqz_x)),
        loss_beta_1 = -tf.reduce_mean(logpx_z + logpz - logqz_x),
        local_logpx_z=tf.reduce_mean(local_logpx_z),
        logpx_z=tf.reduce_mean(logpx_z),
        logpz=tf.reduce_mean(logpz),
        logqz_x=tf.reduce_mean(logqz_x),
        diff_mae=diff_mae,
        local_diff_mae=local_diff_mae)

  def save(self, location):
    self._conditioner.save(location + '/conditioner')
    self._decoder.save(location + '/decoder')
    self._encoder.save(location + '/encoder')
    if self._local_transformation_model is not None:
      self._local_transformation_model.save(location + '/local_transformation_model')

    if self._rotation_model:
      _SaveModel(self._rotation_model, location + '/rotation_model')
  
  def save_weights(self, location):
    self._conditioner.save_weights(location + '/conditioner')
    self._decoder.save_weights(location + '/decoder')
    self._encoder.save_weights(location + '/encoder')
    if self._local_transformation_model is not None:
      self._local_transformation_model.save_weights(location + '/local_transformation_model')
    if self._rotation_model:
      _SaveWeights(self._rotation_model, location + '/rotation_model')

  # TODO: Add support to add  local model.
  def load_model(full_model_location, model_weight_location, local_transformation_model=None,
                 should_load_rotation_model=False):
    return VariationalModel(
        Conditioner(LoadModel(full_model_location, model_weight_location, '/conditioner')),
        Decoder(LoadModel(full_model_location, model_weight_location, '/decoder')),
        Encoder(LoadModel(full_model_location, model_weight_location, '/encoder')),
        LoadModel(full_model_location, model_weight_location, '/rotation_model')
        if should_load_rotation_model else None,
        local_transformation_model)

  def load_weights(self, model_weight_location):
    self._conditioner.load_weights(model_weight_location + '/conditioner')
    self._decoder.load_weights(model_weight_location + '/decoder')
    self._encoder.load_weights(model_weight_location + '/encoder')
    if self._rotation_model:
      self._rotation_model.load_weights(model_weight_location + '/rotation_model')
    self._local_transformation_model.load_weights(model_weight_location+'/local_transformation_model')
