class DecoderTrain(object):
  def __init__(self, model):
    self._model = model

  # Decodes a latent representation to a probability distribution on the
  # location of each atom.
  #
  # Args:
  #  z_0_rescaled: A tenstor for the latent distribution at step 0.
  #    Rescaled by alpha. z_0_rescaled should have dimensions:
  #      (batch_size, num_atoms, num_channels)
  #  cond: A tensor with context information. Should have dimensions:
  #      (batch_size, num_atoms, num_channels)
  #
  # Returns: A multivariate distibution on R^(batch_size X num_atoms)
  def decode(self, z_0_rescaled, cond, training):
    outputs = self._model({
      'z_0_rescaled': z_0_rescaled,
      'cond': cond}, training=training)
    return tfd.MultivariateNormalDiag(
        loc=outputs[0], scale_diag=outputs[1])

  def trainable_weights(self):
    return self._model.trainable_weights

  def save(self, location):
    self._model.save(location, overwrite=True, save_format='tf',
        options=tf.saved_model.SaveOptions())

class EncoderTrain(object):
  def __init__(self, model):
    self._model = model

  # Encodes a latent distribution for each atom.
  #
  # Args:
  #  normalized_coordinates: The atom's coordinates, normalized to have mean 0.
  #  cond: A tensor with context information. Should have dimensions:
  #      (batch_size, num_atoms, num_channels)
  #
  # Returns: A latent distribution with dimensions:
  #   (batch_size, num_atoms, num_channels)
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
  #     Should have shape (batch_size, num_atoms)
  #   atom_names: Integer Tensor representing the atom names of each atom.
  #     Should have shape (batch_size, num_atoms)
  # Returns: The conditioning of the inverse problem. Shoud have shape
  #   (batch_size, num_atoms, num_channels).
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
  #     Should have shape (batch_size, num_atoms, num_channels)
  #   gamma: The value of gamma used in the variance preserving map used to
  #     construct z. Should have shape (batch_size,)
  #   cond: The conditioning passed in to guide the reconstruction.
  #     Should have shape (batch_size, num_atoms, num_channels).
  # Returns: An estimate of the epsilon error introduced. Shoud have shape
  #   (batch_size, num_atoms, num_channels).
  def score(self, z, gamma, cond, training):
    score_val =  self._model({
      'z': z,
      'gamma': gamma,
      'cond': cond}, training=training)
    return score_val

  def trainable_weights(self):
    return self._model.trainable_weights

  def save(self, location):
    self._model.save(location, overwrite=True, save_format='tf', options=tf.saved_model.SaveOptions())

SampleStepWitness = namedtuple(
        'SampleStepWitness', ['alpha', 'ts', 'sigma', 'gamma_t', 'gamma_s', 'eps_avg_mag', 'eps_relative_error', 'error', 'log_prob'])

class DiffusionModel:
  def __init__(self, gamma_module, decoder, encoder, conditioner, scorer):
    self._timesteps = 10000

    self._gamma_module = gamma_module
    self._decoder = decoder
    self._encoder = encoder
    self._conditioner = conditioner
    self._scorer = scorer

  def gamma(self, ts):
    return self._gamma_module.GetGamma(ts)

  def gamma_scalar(self, ts):
    return tf.squeeze(tf.squeeze(self.gamma(tf.expand_dims(tf.expand_dims(ts, -1), -1)), -1), -1)

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
        list(self._gamma_module.trainable_variables))

  def decoder_weights(self):
    return self._decoder.trainable_weights()

  def recon_loss(self, x, f, f_mask, cond, training):
    g_0 = self.gamma_scalar(0.0)
    eps_0 = tf.random.normal(tf.shape(f))
    z_0 = self.variance_preserving_map(f, g_0, eps_0)
    z_0_rescaled = z_0 / self.alpha(g_0)
    prob_dist = self._decoder.decode(z_0_rescaled, cond, training)
    loss_recon = -tf.reduce_sum(
      tf.math.multiply(prob_dist.log_prob(x), f_mask), axis=[-1])
    return loss_recon, tf.reduce_sum(tf.math.multiply(
      tf.math.abs(x-prob_dist.mean()),tf.expand_dims(f_mask, -1)))/tf.reduce_sum(f_mask)

  def latent_loss(self, f, f_mask):
    g_1 = self.gamma_scalar(1.0)
    var_1 = self.sigma2(g_1)
    mean1_sqr = (1. - var_1) * tf.square(f)
    loss_klz = 0.5 * tf.reduce_sum(
    tf.math.multiply(mean1_sqr + var_1 - tf.math.log(var_1) - 1.,
      tf.expand_dims(f_mask, -1)),
      axis=[-1, -2])
    return loss_klz

  def diffusion_loss(self, t, f, f_mask, cond, training):
    # sample z_t.
    g_t = tf.squeeze(self.gamma(tf.expand_dims(t, -1)), -1)
    eps = tf.random.normal(tf.shape(f))
    z_t = self.variance_preserving_map(f, g_t, eps)
    # compute predicted noise
    eps_hat = self._scorer.score(z_t, g_t, cond, training)
    # MSE of predicted noise
    loss_diff_se = tf.reduce_sum(
        tf.math.multiply(tf.square(eps - eps_hat), tf.expand_dims(f_mask, -1)), axis=[-1, -2])
    loss_diff_mse = tf.reduce_sum(loss_diff_se)/tf.reduce_sum(f_mask)

    # loss for finite depth T, i.e. discrete time
    T = self._timesteps
    s = t - (1.0/T)
    g_s = tf.squeeze(self.gamma(tf.expand_dims(s, -1)), -1)
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

  def perfect_score(self, z_t, f, gamma):
    a = self.alpha(gamma)
    var = self.sigma2(gamma)
    return tf.divide(z_t - a*f, tf.math.sqrt(var))

  def perfect_score_vec(self, z_t, f, gamma):
    a = tf.expand_dims(tf.expand_dims(self.alpha(gamma), -1), -1)
    var = tf.expand_dims(tf.expand_dims(self.sigma2(gamma), -1), -1)
    return tf.divide(z_t - a*f, tf.math.sqrt(var))

  def sample_step_vec(self, t, s, z_t, cond, f):
    eps = tf.random.normal(tf.shape(z_t))

    g_s = tf.squeeze(self.gamma(tf.expand_dims(s, -1)), -1)
    g_t = tf.squeeze(self.gamma(tf.expand_dims(t, -1)), -1)
    sigma2_t = tf.expand_dims(tf.expand_dims(self.sigma2(g_t), -1), -1)
    sigma2_s = tf.expand_dims(tf.expand_dims(self.sigma2(g_s), -1), -1)

    sigma_t = tf.math.sqrt(sigma2_t)
    sigma_s = tf.math.sqrt(sigma2_s)

    alpha_t = tf.expand_dims(tf.expand_dims(self.alpha(g_t), -1), -1)
    alpha_s = tf.expand_dims(tf.expand_dims(self.alpha(g_s), -1), -1)

    alpha_t_s = alpha_t / alpha_s
    sigma2_t_s = sigma2_t - tf.math.square(alpha_t_s)*sigma2_s

    sigma2_q = sigma2_t_s * sigma2_s / sigma2_t
    eps_hat_cond = self._scorer.score(z_t, g_t, cond, training=False)
    x = (z_t - sigma_t*eps_hat_cond)/alpha_t
    z_s = ((alpha_t_s * sigma2_s * z_t / sigma2_t) + (alpha_s * sigma2_t_s * x /sigma2_t) +
            tf.math.sqrt(sigma2_q) * eps)
    return z_s

  def sample_step(self, i, T, z_t, cond, f):
    eps = tf.random.normal(tf.shape(z_t))
    t =  tf.cast((T - i) / T, 'float32')
    s = tf.cast((T - i - 1) / T, 'float32')

    g_s = self.gamma_scalar(s)
    g_t = self.gamma_scalar(t)
    sigma2_t = self.sigma2(g_t)
    sigma2_s = self.sigma2(g_s)

    sigma_t = tf.math.sqrt(self.sigma2(g_t))
    sigma_s = tf.math.sqrt(self.sigma2(g_s))

    alpha_t = self.alpha(g_t)
    alpha_s = self.alpha(g_s)

    alpha_t_s = alpha_t/alpha_s
    sigma2_t_s = sigma2_t - tf.math.square(alpha_t_s)*sigma2_s

    sigma2_q = sigma2_t_s * sigma2_s / sigma2_t

    eps_hat_cond = self._scorer.score(z_t, g_t, cond, training=False)

    x = (z_t -sigma_t*eps_hat_cond)/self.alpha(g_t)
    z_s = ((alpha_t_s * sigma2_s * z_t / sigma2_t) + (alpha_s * sigma2_t_s * x /sigma2_t) +
            tf.math.sqrt(sigma2_q) * eps)
    true_eps = self.perfect_score(z_t, f, g_t)
    wt = SampleStepWitness(
            alpha=alpha_t,
            ts = t,
            sigma=sigma_t,
            gamma_t=g_t,
            gamma_s=g_s,
            eps_avg_mag=tf.math.reduce_sum(tf.math.abs(true_eps)),
            eps_relative_error=tf.math.reduce_sum(tf.math.abs(eps_hat_cond-true_eps))/
                               tf.math.reduce_sum(tf.math.abs(true_eps)),
            error=tf.norm(z_s/alpha_s - f),
            log_prob=self.log_prob(f, z_s, g_s))
    return z_s, wt

  def reconstruct(self, t, training_data):
    # Compute x and the conditioning.
    x = training_data['normalized_coordinates']
    cond = self._conditioner.conditioning(
        training_data['residue_names'],
        training_data['atom_names'], training=False)
    # Encode x into the embedding space.
    f = self._encoder.encode(x, cond, training=False)

    # Introduce the error.
    T = self._timesteps
    tn = math.ceil(t * T)
    t = tn / T
    print('t', t)
    g_t = self.gamma_scalar(t)
    eps  = tf.random.normal(tf.shape(f))
    print('true eps', eps)
    z_with_error = self.variance_preserving_map(f, g_t, eps)
    print('f', f)
    z_t = z_with_error
    print('z_t', z_t)

    # Remove the error.
    witnesses = []
    for i in range(T-tn, T):
      if i%100==0:
        print(i)
      z_t, wt = self.sample_step(tf.constant(i), T, z_t, cond, f)
      witnesses.append(wt)

    # Decode from the embedding space.
    g0 = self.gamma_scalar(0)
    z_0_rescaled = z_t /  self.alpha(g0)
    print('z_0', z_t)
    print('z_0_rescaled', z_0_rescaled)
    return (self._decoder.decode(z_with_error / self.alpha(g0), cond, training=False),
        self._decoder.decode(z_0_rescaled, cond, training=False),
        f, z_with_error, z_t, witnesses)

  def generate(self, training_data, embedding_size):
    z_t = tf.random.normal((1, tf.shape(training_data['normalized_coordinates'])[1], embedding_size))

    # Compute f to understand errors during the reverse process.
    x = training_data['normalized_coordinates']
    cond = self._conditioner.conditioning(
        training_data['residue_names'],
        training_data['atom_names'], training=False)
    f = self._encoder.encode(x, cond, training=False)

    # Remove the error.
    T = self._timesteps
    witnesses = []
    for i in range(T):
      if i%100==0:
        print(i)
      z_t, wt = self.sample_step(tf.constant(i), T, z_t, cond, f)
      witnesses.append(wt)

    # Decode from the embedding space
    g0 = self.gamma_scalar(0)
    z_0_rescaled = z_t / self.alpha(g0)
    return (self._decoder.decode(z_0_rescaled, cond, training=False), witnesses)

  # Computes the diffusion loss at multiple timesteps.
  def MSEAtTimesteps(self, ts, training_data):
    x = training_data['normalized_coordinates']
    cond = self._conditioner.conditioning(
        training_data['residue_names'],
        training_data['atom_names'])
    z_0 = self._encoder.encode(x, cond)
    x_mask = tf.cast(tf.math.reduce_any(tf.math.not_equal(x, 0), axis=[-1]), tf.float32)
    for t in ts:
      print(self.diffusion_loss(t, z_0, x_mask, cond))

  def set_scorer(self, scorer):
    self._scorer = scorer

  def set_gamma_module(self, gamma_module):
    self._gamma_module = gamma_module

  def save(self, location):
    self._decoder.save(location + '/decoder_model')
    self._encoder.save(location + '/encoder_model')
    self._conditioner.save(location + '/conditioner_model')
    self._scorer.save(location + '/scorer_model')
    tf.saved_model.save(self._gamma_module, location + '/gamma_module')

  # Computes the PDF for sampling z at time t.
  def log_prob(self, f, z_t, g_t):
    alpha = self.alpha(g_t)
    var = self.sigma2(g_t)

    z_t_distribution = tfd.MultivariateNormalDiag(
        loc=f*alpha, scale_diag=tf.math.sqrt(var)*tf.ones_like(f))
    return tf.math.reduce_sum(z_t_distribution.log_prob(z_t))

def LoadDiffusionModel(location_prefix):
  return DiffusionModel(
      residue_lookup_size, atom_lookup_size,
      tf.saved_model.load(location_prefix + '/gamma_module'),
      DecoderTrain(tf.keras.models.load_model(location_prefix+'/decoder_model')),
      EncoderTrain(tf.keras.models.load_model(location_prefix+'/encoder_model')),
      CondTrain(tf.keras.models.load_model(location_prefix+'/conditioner_model')),
      ScoreTrain(tf.keras.models.load_model(location_prefix+'/scorer_model')))
