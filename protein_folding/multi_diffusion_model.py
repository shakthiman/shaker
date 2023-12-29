import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

class DecoderTrain(object):
  def __init__(self, model):
    self._model = model

  # Decodes a latent representation to a probability distribution on the
  # location of each atom.
  #
  # Args:
  #  z_0_rescaled: A tenstor for the latent distribution at step 0.
  #    Rescaled by alpha. z_0_rescaled should have dimensions:
  #      (batch_size, num_proteins, num_atoms, num_channels)
  #  cond: A tensor with context information. Should have dimensions:
  #      (batch_size, num_proteins, num_atoms, num_channels)
  #
  # Returns: A multivariate distibution on R^(batch_size X num_proteins X num_atoms)
  def decode(self, z_0_rescaled, cond, training):
    outputs = self._model({
      'z_0_rescaled': z_0_rescaled,
      'cond': cond}, training=training)
    return tfd.MultivariateNormalDiag(
        loc=outputs[0], scale_diag=outputs[1])

  def trainable_weights(self):
    return self._model.trainable_weights

  def save(self, tmp_location, location):
    self._model.save_weights(location, overwrite=True, save_format='tf',
        options=tf.saved_model.SaveOptions())

class EncoderTrain(object):
  def __init__(self, model):
    self._model = model

  # Encodes a latent distribution for each atom.
  #
  # Args:
  #  normalized_coordinates: The atom's coordinates, normalized to have mean 0.
  #   Should have dimensions: (batch_size, num_proteins, num_atoms, 3)
  #  cond: A tensor with context information. Should have dimensions:
  #      (batch_size, num_proteins, num_atoms, num_channels)
  #
  # Returns: A latent distribution with dimensions:
  #   (batch_size, num_proteins, num_atoms, num_channels)
  def encode(self, normalized_coordinates, cond, training):
    to_return = self._model({
      'normalized_coordinates': normalized_coordinates,
      'cond': cond}, training=training)
    return to_return

  def trainable_weights(self):
    return self._model.trainable_weights

  def save(self, location):
    self._model.save(location, overwrite=True, save_format='tf',
            options=tf.saved_model.SaveOptions())

class CondTrain(object):
  def __init__(self, model):
    self._model = model

  # Returns a conditioning for the Inverse Problem.
  # Args:
  #   residue_names: Integer Tensor representing the residue names of each atom.
  #     Should have shape (batch_size, num_proteins, num_atoms)
  #   atom_names: Integer Tensor representing the atom names of each atom.
  #     Should have shape (batch_size, num_proteins, num_atoms)
  # Returns: The conditioning of the inverse problem. Shoud have shape
  #   (batch_size, num_proteins, num_atoms, num_channels).
  def conditioning(self, residue_names, atom_names, training):
    return self._model({
      'residue_names': residue_names,
      'atom_names': atom_names}, training=training)

  def trainable_weights(self):
    return self._model.trainable_weights

  def save(self, location):
    self._model.save(
      location, overwrite=True, save_format='tf',
      options=tf.saved_model.SaveOptions())

class ScoreTrain(object):
  def __init__(self, model):
    self._model = model

  # Returns an estimate of the error in z.
  # Args:
  #   z: The latent space embedding with an error introduced.
  #     Should have shape (batch_size, num_proteins, num_atoms, num_channels)
  #   z_mask: A mask with shape
  #     (batch_size, num_proteins, num_atoms, num_channels)
  #   gamma: The value of gamma used in the variance preserving map used to
  #     construct z. Should have shape (batch_size,)
  #   cond: The conditioning passed in to guide the reconstruction.
  #     Should have shape (batch_size, num_proteins, num_atoms, num_channels).
  # Returns: An estimate of the epsilon error introduced. Shoud have shape
  #   (batch_size, num_proteins, num_atoms, num_channels).
  def score(self, z, z_mask, gamma, cond, training):
    score_val =  self._model({
      'z': z,
      'z_mask': z_mask,
      'gamma': gamma,
      'cond': cond}, training=training)
    return score_val

  def trainable_weights(self):
    return self._model.trainable_weights

  def save(self, location):
    self._model.save(location, overwrite=True, save_format='tf', options=tf.saved_model.SaveOptions())

class MultiDiffusionModel:
  def __init__(self, gamma_model, decoder, encoder, conditioner, scorer):
    self._timesteps = 10000

    self._gamma_model = gamma_model
    self._decoder = decoder
    self._encoder = encoder
    self._conditioner = conditioner
    self._scorer = scorer

  def gamma(self, ts):
    return self._gamma_model(ts)

  def gamma_scalar(self, ts):
    return tf.squeeze(self.gamma(tf.expand_dims(ts, -1)), -1)

  def sigma2(self, gamma):
    return tf.math.sigmoid(-1*gamma)

  def alpha(self, gamma):
    return tf.math.sqrt(1-self.sigma2(gamma))

  def variance_preserving_map(self, x, gamma, eps):
    a = self.alpha(gamma)
    var = self.sigma2(gamma)
    s1 = tf.expand_dims(tf.expand_dims(a, axis=-1), axis=-1)*x
    s2 =  (
      tf.expand_dims(
        tf.expand_dims(tf.math.sqrt(var), axis=-1), axis=-1) * eps)
    return s1 + s2

  def trainable_weights(self):
    return (self._decoder.trainable_weights() +
        self._encoder.trainable_weights() +
        self._conditioner.trainable_weights() +
        self._scorer.trainable_weights() +
        list(self._gamma_model.trainable_variables))

  def decoder_weights(self):
    return self._decoder.trainable_weights()

  def recon_loss(self, x, f, f_mask, cond, training):
    g_0 = self.gamma_scalar(0.0)
    eps_0 = tf.random.normal(tf.shape(f))
    z_0 = self.variance_preserving_map(f, g_0, eps_0)
    z_0_rescaled = z_0 / self.alpha(g_0)
    prob_dist = self._decoder.decode(z_0_rescaled, cond, training)
    loss_recon = -tf.reduce_sum(
      tf.math.multiply(prob_dist.log_prob(x), f_mask), axis=[-1, -2])
    return loss_recon, tf.reduce_sum(tf.math.multiply(
      tf.math.abs(x-prob_dist.mean()),tf.expand_dims(f_mask, -1)))/tf.reduce_sum(f_mask)

  def latent_loss(self, f, f_mask):
    g_1 = self.gamma_scalar(1.0)
    var_1 = self.sigma2(g_1)
    mean1_sqr = (1. - var_1) * tf.square(f)
    loss_klz = 0.5 * tf.reduce_sum(
    tf.math.multiply(mean1_sqr + var_1 - tf.math.log(var_1) - 1.,
      tf.expand_dims(f_mask, -1)),
      axis=[-1, -2, -3])
    return loss_klz

  def diffusion_loss(self, t, f, f_mask, cond, training):
    # sample z_t.
    g_t = self.gamma(t)
    eps = tf.random.normal(tf.shape(f))
    z_t = self.variance_preserving_map(f, g_t, eps)
    # compute predicted noise
    eps_hat = self._scorer.score(z_t, f_mask, g_t, cond, training)
    # MSE of predicted noise
    loss_diff_se = tf.reduce_sum(
        tf.math.multiply(tf.square(eps - eps_hat), tf.expand_dims(f_mask, -1)), axis=[-1, -2, -3])
    loss_diff_mse = tf.reduce_sum(loss_diff_se)/tf.reduce_sum(f_mask)

    # loss for finite depth T, i.e. discrete time
    T = self._timesteps
    s = t - (1.0/T)
    g_s = self.gamma(s)
    loss_diff = 0.5 * T * tf.math.expm1(g_s - g_t) * loss_diff_se
    return loss_diff, loss_diff_mse

  def compute_model_loss(self, training_data, training=True):
    x = training_data['normalized_coordinates']
    cond = self._conditioner.conditioning(
        training_data['residue_names'],
        training_data['atom_names'], training)

    n_batch = tf.shape(x)[0]

    # 1. RECONSTRUCTION LOSS
    # add noise and reconstruct
    f = self._encoder.encode(x, cond, training)
    x_mask = tf.cast(
      tf.math.reduce_any(
        tf.math.greater(tf.math.abs(x), 1e-6), axis=[-1]), tf.float32)
    loss_recon, recon_diff = self.recon_loss(x, f, x_mask, cond, training)

    # 2. LATENT LOSS
    # KL z1 with N(0,1) prior
    loss_klz = self.latent_loss(f, x_mask)

    # 3. Diffusion Loss.
    # Sample time steps.
    # Use anithetic time sampling.
    t0 = tf.random.uniform(shape=[])
    t = tf.math.floormod(t0 + tf.range(0, 1, 1./tf.cast(n_batch, 'float32'),
      dtype='float32'), 1.0)

    # Discretize timesteps.
    T = self._timesteps
    t = tf.math.ceil(t*T) / T

    loss_diff, loss_diff_mse = self.diffusion_loss(t, f, x_mask, cond, training)
    return (loss_diff, loss_klz, loss_recon, loss_diff_mse, recon_diff)
