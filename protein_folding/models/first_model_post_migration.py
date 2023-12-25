import tensorflow as tf

from protein_folding import diffusion_model

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

  scale_diag = tf.Variable(1.0)
  loc = tf.keras.layers.Dense(3)(tf.keras.layers.Dense(100, 'gelu')(tf.keras.layers.Dense(100, 'gelu')(concatenated_inputs)))
  return tf.keras.Model(
      inputs=[z_0_rescaled, cond],
      outputs=[loc, tf.keras.layers.Identity()(scale_diag*tf.ones_like(loc))])

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


  return tf.keras.Model(inputs=[normalized_coordinates, cond],
      outputs=tf.keras.layers.Identity()(concatenate_inputs))

def CondModel(residue_lookup_size, atom_lookup_size):
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

  # Reduce, Attend, and Upsample.
  sequence_size = tf.shape(concatenated_features)[1]
  ideal_sequence_size = tf.cast(tf.math.ceil(tf.cast(sequence_size, tf.float32)/10)*10, tf.int32)
  paddings = tf.stack([tf.constant([0, 0]),
                       tf.stack([tf.constant(0), ideal_sequence_size-sequence_size]),
                       tf.constant([0, 0])])
  padded_features = tf.ensure_shape(tf.pad(concatenated_features, paddings), [None,None,161])
  reduced_features = tf.keras.layers.SeparableConv1D(
      64, 10, padding='same')(padded_features)
  reduced_features  = tf.keras.layers.AveragePooling1D(10, padding='same')(reduced_features)
  transformer_output = TransformerLayer(1, 5, 5, 64, reduced_features)
  upsampled_transformer_output = tf.keras.layers.UpSampling1D(10)(
      transformer_output)

  score = tf.keras.layers.Dense(Z_EMBEDDING_SIZE)(tf.keras.layers.concatenate(
      inputs=[concatenated_features, upsampled_transformer_output[:,:sequence_size,:]]))

  return tf.keras.Model(inputs=[z, gamma, cond], outputs=score)

def GammaModel():
  ts = tf.keras.Input(shape=(None, None, 1))
  l1 = tf.keras.layers.Dense(
      1, kernel_constraint=tf.keras.constraints.NonNeg())
  l2 = tf.keras.layers.Dense(
      4096, activation='sigmoid',
      kernel_constraint=tf.keras.constraints.NonNeg())
  l3 = tf.keras.layers.Dense(
      1, kernel_constraint=tf.keras.constraints.NonNeg())
  gamma = -1 * (l1(ts) + l3(l2(ts)))
  return tf.keras.Model(inputs=ts, outputs=gamma)

MODEL_FOR_TRAINING = lambda vocab: diffusion_model.DiffusionModel(
    GammaModel(), diffusion_model.DecoderTrain(DecoderModel()),
    diffusion_model.EncoderTrain(EncoderModel()),
    diffusion_model.CondTrain(
      CondModel(vocab.ResidueLookupSize(), vocab.AtomLookupSize())),
    diffusion_model.ScoreTrain(ScoreModel()))
