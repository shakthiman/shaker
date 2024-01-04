import tensorflow as tf

from protein_folding import multi_diffusion_model

# Constants
Z_EMBEDDING_SIZE = 61
COND_EMBEDDING_SIZE = 6
ENCODER_CONVOLVE_SIZE = 10
TIMESTEP_EMBEDDING_DIMS = 10
AMINO_ACID_EMBEDDING_DIMS = 20

class CustomSelfAttention(tf.keras.layers.Layer):
  def __init__(self, num_heads, key_dim):
    super(CustomSelfAttention, self).__init__()
    self._attention_layer = tf.keras.layers.MultiHeadAttention(num_heads, key_dim)

  def call(self, input_tensor, input_mask):
    def _test_fn(x):
      return self._attention_layer(x[0], x[0], x[0],
          tf.math.logical_and(
            tf.expand_dims(x[1], -1),
            tf.expand_dims(x[1], -2))
          )
    return tf.map_fn(
        _test_fn, tf.tuple([input_tensor, input_mask]),
        fn_output_signature=tf.float32)

# Transformer Unit
def ShapeList(x):
  ps = x.get_shape().as_list()
  ts = tf.shape(x)
  return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def RefactorX(x, n):
  x_shape = ShapeList(x)
  assert len(x_shape) == 3, len(x_shape)
  return tf.reshape(x, [
    x_shape[0], # Batch size remains the same.
    -1, # Additional blocks are introduced.
    x_shape[1] // n, # Number of timesteps reduced by a factor of n.
    x_shape[2] # Embedding dimension does not change.
    ])

def RefactorXMask(x_mask, n):
  x_mask_shape = ShapeList(x_mask)
  assert len(x_mask_shape) == 2, len(x_mask_shape)
  return tf.reshape(x_mask, [
    x_mask_shape[0], # Batch size remains the same.
    -1, # Additional blocks are introduced.
    x_mask_shape[1] // n # Number of timesteps reduced by a factor of n.
    ])


def StraightenMultipeptideSequence(x):
  x_shape = ShapeList(x)
  assert len(x_shape) == 4
  return tf.reshape(x,[
    x_shape[0], # Batch size remains the same.
    -1, # Amino acid dimension.
    x_shape[3]])

def StraightenMultipeptideMask(x):
  x_shape = ShapeList(x)
  assert len(x_shape) == 3
  return tf.reshape(x, [
    x_shape[0], # Batch size remains the same.
    -1 # Amino Acid dimension.
    ])

def TransposeAndAttend(attention_layer, refactored_x, refactored_mask, perm):
  transposed_x = tf.transpose(refactored_x, perm)
  transposed_mask = tf.transpose(refactored_mask, perm[:-1])
  score = attention_layer(transposed_x, transposed_mask)
  return tf.transpose(score, perm)

def AttentionLayer(num_blocks, num_heads, key_dim, inputs, inputs_mask):
  refactored_x = RefactorX(inputs, num_blocks)
  refactored_mask = RefactorXMask(tf.cast(inputs_mask, tf.bool), num_blocks)
  local_self_attention = CustomSelfAttention(num_heads, key_dim)(refactored_x, refactored_mask)
  global_self_attention = TransposeAndAttend(CustomSelfAttention(num_heads, key_dim), refactored_x, refactored_mask, [0, 2, 1, 3])
  return tf.keras.layers.LayerNormalization()(
      tf.keras.layers.Add()([
          inputs,
          tf.reshape(local_self_attention, ShapeList(inputs)),
          tf.reshape(global_self_attention, ShapeList(inputs))]))

def FeedForwardLayer(num_layers, output_size, inputs):
  t = inputs
  for i in range(num_layers):
    t = tf.keras.layers.Dense(100, 'gelu')(t)
  t = tf.keras.layers.Dense(output_size)(t)
  return tf.keras.layers.LayerNormalization()(
      tf.keras.layers.Add()([inputs, t]))

def TransformerLayer(num_transformers, num_blocks, num_heads, key_dim, num_dnn_layers, output_size,
    inputs, inputs_mask):
  x = inputs
  for i in range(num_transformers):
    x = AttentionLayer(num_blocks, num_heads, key_dim, x, inputs_mask)
    x = FeedForwardLayer(num_dnn_layers, output_size, x)
  return x

def AminoAcidPositionalEmbedding(z):
  pos_indices = tf.expand_dims(
      tf.expand_dims(tf.expand_dims(
        tf.range(tf.shape(z)[2], dtype='float32'), 0), 0) *
      tf.ones(tf.shape(z)[:-1]), -1)
  half_dim = AMINO_ACID_EMBEDDING_DIMS // 2
  pemb = tf.math.log(20_000.0) / (half_dim - 1)
  pemb = tf.math.exp(tf.range(half_dim, dtype='float') * - pemb)
  pemb = pos_indices * pemb
  pemb = tf.keras.layers.concatenate([tf.math.sin(pemb), tf.math.cos(pemb)])
  return pemb

class VectorizedMapLayer(tf.keras.layers.Layer):
  def __init__(self, map_layer):
    super(VectorizedMapLayer, self).__init__()
    self._map_layer = map_layer

  def call(self, input_tensor):
    return tf.vectorized_map(self._map_layer, input_tensor)

def EncoderModel():
  normalized_coordinates = tf.keras.Input(shape=(None, None, 3),
      name='normalized_coordinates')
  cond = tf.keras.Input(shape=(None, None, COND_EMBEDDING_SIZE), name='cond')

  pemb = AminoAcidPositionalEmbedding(normalized_coordinates)

  encoded_coordinates = tf.keras.layers.concatenate(
      inputs=[normalized_coordinates, cond, pemb])

  conv_layer = tf.keras.layers.SeparableConv1D(
      32, ENCODER_CONVOLVE_SIZE, activation='gelu', padding='same')
  convolved_coordinates = VectorizedMapLayer(conv_layer)(encoded_coordinates)
  concatenate_inputs = tf.keras.layers.concatenate(inputs=[
    convolved_coordinates, encoded_coordinates])

  return tf.keras.Model(inputs=[normalized_coordinates, cond],
      outputs=tf.keras.layers.Identity()(concatenate_inputs))

def DecoderModel():
  z_0_rescaled = tf.keras.Input(
      shape=(None, None, Z_EMBEDDING_SIZE),
      name='z_0_rescaled')
  cond = tf.keras.Input(shape=(None, None, COND_EMBEDDING_SIZE), name='cond')
  pemb = AminoAcidPositionalEmbedding(z_0_rescaled)
  
  base_inputs = tf.keras.layers.concatenate(inputs=[
    z_0_rescaled, cond, pemb])
  conv_layer = tf.keras.layers.Conv1DTranspose(32, ENCODER_CONVOLVE_SIZE, padding='same')
  convolved_inputs = VectorizedMapLayer(conv_layer)(base_inputs)
  concatenated_inputs = tf.keras.layers.concatenate(inputs=[base_inputs, convolved_inputs])

  scale_diag = tf.Variable(1.0)
  loc = tf.keras.layers.Dense(3)(tf.keras.layers.Dense(100, 'gelu')(tf.keras.layers.Dense(100, 'gelu')(concatenated_inputs)))
  return tf.keras.Model(
      inputs=[z_0_rescaled, cond],
      outputs=[loc, tf.keras.layers.Identity()(scale_diag*tf.ones_like(loc))])

def CondModel(residue_lookup_size, atom_lookup_size):
  residue_names = tf.keras.Input(shape=(None, None), name='residue_names')
  atom_names = tf.keras.Input(shape=(None, None), name='atom_names')

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
  z = tf.keras.Input(shape=(None, None, Z_EMBEDDING_SIZE), name='z')
  z_mask = tf.keras.Input(shape=(None, None), name='z_mask')
  gamma = tf.keras.Input(shape=[], name='gamma')
  cond = tf.keras.Input(shape=(None, None, COND_EMBEDDING_SIZE), name='cond')

  # Compute timestep embedding
  t = gamma * 1000
  t = tf.expand_dims(tf.expand_dims(tf.expand_dims(t, -1), -1)
      * tf.ones(tf.shape(z)[:-1]), -1)
  half_dim = TIMESTEP_EMBEDDING_DIMS // 2
  temb = tf.math.log(10_000.0) / (half_dim - 1)
  temb = tf.math.exp(tf.range(half_dim, dtype='float') * - temb)
  temb = t * temb
  temb = tf.keras.layers.concatenate([tf.math.sin(temb), tf.math.cos(temb)])

  # Compute Amino Acid Positional Embedding
  pemb = AminoAcidPositionalEmbedding(z)

  num_blocks = 200
  base_features = tf.keras.layers.concatenate(
      inputs=[z, cond, temb, pemb])
  score_convolve_layer = tf.keras.layers.Conv1DTranspose(
      64, ENCODER_CONVOLVE_SIZE, padding='same', activation='gelu')
  convolved_features = VectorizedMapLayer(score_convolve_layer)(base_features)
  concatenated_features = tf.keras.layers.concatenate(
      inputs=[base_features, convolved_features])

  # Reduce, Attend, and Upsample.
  original_shape = ShapeList(concatenated_features)
  assert len(original_shape) == 4
  straightened_features = StraightenMultipeptideSequence(concatenated_features)
  straightened_mask = StraightenMultipeptideMask(z_mask)

  sequence_size = ShapeList(straightened_features)[1]
  # Pool the features.
  ideal_sequence_size = tf.cast(
      tf.math.ceil(
        tf.cast(sequence_size, tf.float32)/num_blocks)*num_blocks, tf.int32)
  paddings = tf.stack([tf.constant([0, 0]),
                       tf.stack([tf.constant(0), ideal_sequence_size-sequence_size]),
                       tf.constant([0, 0])])
  padded_features = tf.pad(straightened_features, paddings)
  padded_features = tf.ensure_shape(padded_features, [None, None, 161])
  padded_mask = tf.pad(straightened_mask, tf.stack([
    tf.constant([0, 0]),
    tf.stack([tf.constant(0), ideal_sequence_size-sequence_size])]))

  transformer_output = TransformerLayer(10, num_blocks, 5, 10, 5, 161, padded_features, padded_mask)
  transformer_output = transformer_output[:,:sequence_size,:]
  transformer_output = tf.reshape(transformer_output,
      [original_shape[0], original_shape[1], original_shape[2], -1])
  
  score = tf.keras.layers.Dense(Z_EMBEDDING_SIZE)(tf.keras.layers.concatenate(
    inputs=[concatenated_features, tf.ensure_shape(transformer_output, [None, None, None, 161])]))

  return tf.keras.Model(inputs=[z, z_mask, gamma, cond], outputs=score)

def GammaModel():
  ts = tf.keras.Input(shape=(None, None))
  expanded_ts = tf.expand_dims(ts, -1)
  l1 = tf.keras.layers.Dense(
      1, kernel_constraint=tf.keras.constraints.NonNeg())
  l2 = tf.keras.layers.Dense(
      2048, activation='sigmoid',
      kernel_constraint=tf.keras.constraints.NonNeg())
  l3 = tf.keras.layers.Dense(
      1, kernel_constraint=tf.keras.constraints.NonNeg())
  gamma = -1 * (l1(expanded_ts) + l3(l2(expanded_ts)) + ts)
  return tf.keras.Model(inputs=ts, outputs=gamma)

MODEL_FOR_TRAINING = lambda vocab: multi_diffusion_model.MultiDiffusionModel(
    GammaModel(), multi_diffusion_model.DecoderTrain(DecoderModel()),
    multi_diffusion_model.EncoderTrain(EncoderModel()),
    multi_diffusion_model.CondTrain(
      CondModel(vocab.ResidueLookupSize(), vocab.AtomLookupSize())),
    multi_diffusion_model.ScoreTrain(ScoreModel()))
