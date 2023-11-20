def GetStaticVocab(vocab):
  return tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(
        tf.constant(vocab),
        tf.constant(range(2, len(vocab)+2), dtype=tf.int64),
        key_dtype=tf.string,
        value_dtype=tf.int64), 1)

# Getting the Residue and Vocab names from training_examples_summary.
client = storage.Client('shaker-388116')
bucket = client.bucket('unreplicated-training-data')
blob = bucket.get_blob('training_examples_summary/data-00000-of-00001.avro')
reader = datafile.DataFileReader(blob.open('rb'), io.DatumReader())
residue_names = []
atom_names = []
for x in reader:
  residue_names = x['residue_names']
  atom_names = x['atom_names']
with tf.device('/CPU:0'):
  residue_names_preprocessor = GetStaticVocab(residue_names)
  atom_names_preprocessor = GetStaticVocab(atom_names)

# Convert from training example to TF Features.
raw_tf_examples = tf.data.TFRecordDataset(
    ['gs://unreplicated-training-data/training_examples/data-{0}-of-00037.tfrecord'.format(str(i).zfill(5))
    for i in range(37)])
feature_spec = {
  'name': tf.io.FixedLenFeature([], tf.string, default_value=''),
  'residue_names': tf.io.FixedLenFeature([], tf.string, default_value=''),
  'atom_names': tf.io.FixedLenFeature([], tf.string, default_value=''),
  'normalized_coordinates': tf.io.FixedLenFeature([], tf.string, default_value='')}


def _ConvertToFeatures(
    x, feature_spec, residue_names_preprocessor, atom_names_preprocessor):
  example = tf.io.parse_single_example(x, feature_spec)
  residue_names = residue_names_preprocessor.lookup(
      tf.io.parse_tensor(example['residue_names'], tf.string))
  atom_names = atom_names_preprocessor.lookup(
      tf.io.parse_tensor(example['atom_names'], tf.string))
  normalized_coordinates = tf.io.parse_tensor(
    example['normalized_coordinates'], tf.float32)
  return {
    'name': example['name'],
    'residue_names': residue_names,
    'atom_names': atom_names,
    'normalized_coordinates': normalized_coordinates}
ds = raw_tf_examples.map(lambda x: _ConvertToFeatures(
  x, feature_spec, residue_names_preprocessor, atom_names_preprocessor))

## Residue and Atom lookup size
residue_lookup_size = len(residue_names) + 2
atom_lookup_size = len(atom_names) + 2


# Constants
Z_EMBEDDING_SIZE = 61
COND_EMBEDDING_SIZE = 6
ENCODER_CONVOLVE_SIZE = 10
TIMESTEP_EMBEDDING_DIMS = 10
AMINO_ACID_EMBEDDING_DIMS = 20

# Transformer Unit
def AttentionLayer(num_heads, inputs):
  return tf.keras.layers.LayerNormalization()(
      tf.keras.layers.Add()([
        inputs, tf.keras.layers.MultiHeadAttention(num_heads, 10)(
          inputs, inputs, inputs)]))

def FeedForwardLayer(num_layers, output_size, inputs):
  t = inputs
  for i in range(num_layers):
    t = tf.keras.layers.Dense(100, 'gelu')(t)
  t = tf.keras.layers.Dense(output_size)(t)
  return tf.keras.layers.LayerNormalization()(
      tf.keras.layers.Add()([inputs, t]))

def TransformerLayer(num_transformers, num_heads, num_dnn_layers, output_size, inputs):
  x = inputs
  for i in range(num_transformers):
    x = AttentionLayer(num_heads, x)
    x = FeedForwardLayer(num_dnn_layers, output_size, x)
  return x

def PositionalEmbedding(z):
  pos_indices = tf.expand_dims(
      tf.expand_dims(
        tf.range(tf.shape(z)[1], dtype='float32'), 0) *
      tf.ones(tf.shape(z)[:-1]), -1)
  half_dim = AMINO_ACID_EMBEDDING_DIMS // 2
  pemb = tf.math.log(20_000.0) / (half_dim - 1)
  pemb = tf.math.exp(tf.range(half_dim, dtype='float') * - pemb)
  pemb = pos_indices * pemb
  pemb = tf.keras.layers.concatenate([tf.math.sin(pemb), tf.math.cos(pemb)])
  return pemb

def DecoderModel():
  z_0_rescaled = tf.keras.Input(
      shape=(None, Z_EMBEDDING_SIZE),
      name='z_0_rescaled')
  cond = tf.keras.Input(shape=(None, COND_EMBEDDING_SIZE), name='cond')
  pemb = PositionalEmbedding(z_0_rescaled)
  
  base_inputs = tf.keras.layers.concatenate(inputs=[
    z_0_rescaled, cond, pemb])
  convolved_inputs = tf.keras.layers.Conv1DTranspose(32, ENCODER_CONVOLVE_SIZE, padding='same')(base_inputs)
  concatenated_inputs = tf.keras.layers.concatenate(inputs=[base_inputs, convolved_inputs])
  transformer_output = TransformerLayer(1, 5, 5, 119, concatenated_inputs)

  scale_diag = tf.Variable(1.0)
  loc = tf.keras.layers.Dense(3)(tf.keras.layers.Dense(100, 'gelu')(tf.keras.layers.Dense(100, 'gelu')(transformer_output)))
  return tf.keras.Model(
      inputs=[z_0_rescaled, cond],
      outputs=[loc, tf.keras.layers.Identity()(scale_diag*tf.ones_like(loc))])

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

def EncoderModel():
  normalized_coordinates = tf.keras.Input(shape=(None, 3),
      name='normalized_coordinates')
  cond = tf.keras.Input(shape=(None, COND_EMBEDDING_SIZE), name='cond')

  pemb = PositionalEmbedding(normalized_coordinates)

  encoded_coordinates = tf.keras.layers.concatenate(
      inputs=[normalized_coordinates, cond, pemb])


  convolved_coordinates = tf.keras.layers.SeparableConv1D(
      32, ENCODER_CONVOLVE_SIZE, activation='gelu', padding='same')(encoded_coordinates)
  concatenate_inputs = tf.keras.layers.concatenate(inputs=[
    convolved_coordinates, encoded_coordinates])
  transformer_output = TransformerLayer(1, 5, 5, 61, concatenate_inputs)



  return tf.keras.Model(inputs=[normalized_coordinates, cond],
      outputs=tf.keras.layers.Identity()(transformer_output))

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

def CondModel():
  residue_names = tf.keras.Input(shape=(None,), name='residue_names')
  atom_names = tf.keras.Input(shape=(None,), name='atom_names')

  residue_embeddings = tf.keras.layers.Embedding(
    input_dim=residue_lookup_size,
    output_dim=3)(residue_names)
  atom_embeddings = tf.keras.layers.Embedding(
    input_dim=atom_lookup_size,
    output_dim=3)(atom_names)

  cond_out = tf.keras.layers.concatenate(
    inputs=[residue_embeddings, atom_embeddings])
  cond_out = tf.keras.layers.Dense(100, 'gelu')(cond_out)
  cond_out = tf.keras.layers.Dense(COND_EMBEDDING_SIZE)(cond_out)

  return tf.keras.Model(inputs=[residue_names, atom_names], outputs=cond_out)

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

def ScoreModel():
  z = tf.keras.Input(shape=(None, Z_EMBEDDING_SIZE), name='z')
  gamma = tf.keras.Input(shape=[], name='gamma')
  cond = tf.keras.Input(shape=(None, COND_EMBEDDING_SIZE), name='cond')

  # Compute timestep embedding
  t = gamma * 1000
  t = tf.expand_dims(tf.expand_dims(t, -1) * tf.ones(tf.shape(z)[:-1]), -1)
  half_dim = TIMESTEP_EMBEDDING_DIMS // 2
  temb = tf.math.log(10_000.0) / (half_dim - 1)
  temb = tf.math.exp(tf.range(half_dim, dtype='float') * - temb)
  temb = t * temb
  temb = tf.keras.layers.concatenate([tf.math.sin(temb), tf.math.cos(temb)])

  # Compute Amino Acid Positional Embedding
  pos_indices = tf.expand_dims(tf.expand_dims(tf.range(tf.shape(z)[1], dtype='float32'), 0) * tf.ones(tf.shape(z)[:-1]), -1)
  half_dim = AMINO_ACID_EMBEDDING_DIMS // 2
  pemb = tf.math.log(20_000.0) / (half_dim - 1)
  pemb = tf.math.exp(tf.range(half_dim, dtype='float') * - pemb)
  pemb = pos_indices * pemb
  pemb = tf.keras.layers.concatenate([tf.math.sin(pemb), tf.math.cos(pemb)])

  base_features = tf.keras.layers.concatenate(
      inputs=[z, cond, temb, pemb])
  score_convolve_layer = tf.keras.layers.Conv1DTranspose(
      64, ENCODER_CONVOLVE_SIZE, padding='same', activation='gelu')
  concatenated_features = tf.keras.layers.concatenate(
      inputs=[base_features, score_convolve_layer(base_features)])

  transformer_output = TransformerLayer(1, 5, 5, 161, concatenated_features)
  score = tf.keras.layers.Dense(Z_EMBEDDING_SIZE)(transformer_output)

  return tf.keras.Model(inputs=[z, gamma, cond], outputs=score)

def PerfectScoreModel(perfect_knowledge):
  z = tf.keras.Input(shape=(None, Z_EMBEDDING_SIZE), name='z')
  gamma = tf.keras.Input(shape=[], name='gamma')
  cond = tf.keras.Input(shape=(None, COND_EMBEDDING_SIZE), name='cond')

  a = tf.math.sqrt(1 - tf.math.sigmoid(-1*gamma))
  var = tf.math.sigmoid(-1*gamma)
  score = tf.divide((z - a*perfect_knowledge), tf.math.sqrt(var))
  return tf.keras.Model(inputs=[z,gamma,cond], outputs=score)

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

class GammaModule(tf.Module):
  def __init__(self):
    self._l1 = tf.keras.layers.Dense(
      1, kernel_constraint=tf.keras.constraints.NonNeg())
    self._l2 = tf.keras.layers.Dense(
        1024, activation='sigmoid',
        kernel_constraint=tf.keras.constraints.NonNeg())
    self._l3 = tf.keras.layers.Dense(
        1, kernel_constraint=tf.keras.constraints.NonNeg())

  @tf.function
  def GetGamma(self, ts):
    l1_t = self._l1(ts)
    return -1*(l1_t + self._l3(self._l2(self._l1(ts))))

class DiffusionModel:
  def __init__(self, residue_lookup_size, atom_lookup_size,
      gamma_module, decoder, encoder, conditioner, scorer):
    self._timesteps = 10000

    self._gamma_module = gamma_module
    self._decoder = decoder
    self._encoder = encoder
    self._conditioner = conditioner
    self._scorer = scorer

  def gamma(self, ts):
    return self._gamma_module.GetGamma(ts)

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
    g_0 = self.gamma(tf.constant([[0]]))[0][0]
    eps_0 = tf.random.normal(tf.shape(f))
    z_0 = self.variance_preserving_map(f, g_0, eps_0)
    z_0_rescaled = z_0 / self.alpha(g_0)
    prob_dist = self._decoder.decode(z_0_rescaled, cond, training)
    loss_recon = -tf.reduce_sum(
      tf.math.multiply(prob_dist.log_prob(x), f_mask), axis=[-1])
    return loss_recon, tf.reduce_sum(tf.math.multiply(
      tf.math.abs(x-prob_dist.mean()),tf.expand_dims(f_mask, -1)))/tf.reduce_sum(f_mask)

  def latent_loss(self, f, f_mask):
    g_1 = self.gamma(tf.constant([[1.0]]))[0][0]
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

  @tf.function(reduce_retracing=True)
  def sample_step(self, i, T, z_t, cond):
    eps = tf.random.normal(tf.shape(z_t))
    eps = tf.zeros(tf.shape(z_t))
    t =  tf.cast((T - i) / T, 'float32')
    s = tf.cast((T - i - 1) / T, 'float32')

    g_s = self.gamma(s)
    g_t = self.gamma(t)
    sigma2_t = self.sigma2(g_t)
    sigma2_s = self.sigma2(g_s)

    sigma_t = tf.math.sqrt(self.sigma2(g_t))
    sigma_s = tf.math.sqrt(self.sigma2(g_s))

    alpha_t = self.alpha(g_t)
    alpha_s = self.alpha(g_s)

    alpha_t_s = alpha_t/alpha_s
    sigma2_t_s = sigma2_t - tf.math.square(alpha_t_s)*sigma2_s

    eps_hat_cond = self._scorer.score(z_t, g_t, cond, training=False)

    x = (z_t -sigma_t*eps_hat_cond)/self.alpha(g_t)
    z_s = (alpha_t_s * sigma2_s * z_t / sigma2_t) + (alpha_s * sigma2_t_s * x /sigma2_t)
    return z_s

  def reconstruct(self, t, training_data):
    # Compute x and the conditioning.
    x = training_data['normalized_coordinates']
    cond = self._conditioner.conditioning(
        training_data['residue_names'],
        training_data['atom_names'], training=False)
    # Encode x into the embedding space.
    z_0 = self._encoder.encode(x, cond, training=False)

    # Introduce the error.
    T = self._timesteps
    tn = math.ceil(t * T)
    t = tn / T
    print('t', t)
    g_t = self.gamma(t)
    eps  = tf.random.normal(tf.shape(z_0))
    print('true eps', eps)
    z_with_error = self.variance_preserving_map(z_0, g_t, eps)
    print('z_0', z_0)
    z_t = z_with_error
    print('z_t', z_t)

    # Remove the error.
    for i in range(T-tn, T):
      if i%100==0:
        print(i)
      z_t = self.sample_step(tf.constant(i), T, z_t, cond)

    # Decode from the embedding space.
    g0 = self.gamma(0)
    z_0_rescaled = z_t /  self.alpha(g0)
    print('z_0', z_t)
    print('z_0_rescaled', z_0_rescaled)
    return (self._decoder.decode(z_with_error / self.alpha(g0), cond, training=False),
        self._decoder.decode(z_0_rescaled, cond, training=False),
        z_0, z_with_error, z_t)

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

def LoadDiffusionModel(location_prefix):
  return DiffusionModel(
      residue_lookup_size, atom_lookup_size,
      tf.saved_model.load(location_prefix + '/gamma_module'),
      DecoderTrain(tf.keras.models.load_model(location_prefix+'/decoder_model')),
      EncoderTrain(tf.keras.models.load_model(location_prefix+'/encoder_model')),
      CondTrain(tf.keras.models.load_model(location_prefix+'/conditioner_model')),
      ScoreTrain(tf.keras.models.load_model(location_prefix+'/scorer_model')))
