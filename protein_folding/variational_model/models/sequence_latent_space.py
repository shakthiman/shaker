import tensorflow as tf

from protein_folding.variational_model import model

_COND_EMBEDDING_SIZE = 6
_LATENT_EMBEDDING_SIZE = 3
_AMINO_ACID_EMBEDDING_DIMS = 20

def ShapeList(x):
  ps = x.get_shape().as_list()
  ts = tf.shape(x)
  return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def AminoAcidPositionalEmbedding(cond):
  pos_indices = tf.expand_dims(
      tf.expand_dims(tf.expand_dims(
        tf.range(tf.shape(cond)[2], dtype='float32'), 0), 0) *
      tf.ones(tf.shape(cond)[:-1]), -1)
  half_dim = _AMINO_ACID_EMBEDDING_DIMS // 2
  pemb = tf.math.log(20_000.0) / (half_dim - 1)
  pemb = tf.math.exp(tf.range(half_dim, dtype='float') * - pemb)
  pemb = pos_indices * pemb
  pemb = tf.keras.layers.concatenate([tf.math.sin(pemb), tf.math.cos(pemb)])
  return pemb

def PeptideIndx(cond):
  pos_indices = tf.expand_dims(
      tf.expand_dims(tf.expand_dims(
        tf.range(tf.shape(cond)[1], dtype='float32'), 0), -1) *
      tf.ones(tf.shape(cond)[:-1]), -1)
  return pos_indices

def StraightenMultipeptideSequence(x):
  x_shape = ShapeList(x)
  assert len(x_shape) == 4
  return tf.reshape(x,[
    x_shape[0], # Batch size remains the same.
    -1, # Amino acid dimension.
    x_shape[3]])

def StraightenMultipeptideMask(atom_mask):
  mask_shape = ShapeList(atom_mask)
  assert len(mask_shape) == 3
  return tf.reshape(atom_mask, [
    mask_shape[0], # Batch size remains the same.
    -1 # Amino Acid dimension.
    ])

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

class CustomSelfAttention(tf.keras.layers.Layer):
  def __init__(self, num_heads, key_dim):
    super(CustomSelfAttention, self).__init__()
    self._attention_layer = tf.keras.layers.MultiHeadAttention(num_heads, key_dim)

  def call(self, input_tensor, input_mask):
    def _apply_attention(x):
      return self._attention_layer(x[0], x[0], x[0],
          tf.math.logical_and(
            tf.expand_dims(x[1], -1),
            tf.expand_dims(x[1], -2))
          )
    return tf.map_fn(
        _apply_attention, tf.tuple([input_tensor, input_mask]),
        fn_output_signature=tf.float32)

def TransposeAndAttend(attention_layer, refactored_x, refactored_mask, perm):
  transposed_x = tf.transpose(refactored_x, perm)
  transposed_mask = tf.transpose(refactored_mask, perm[:-1])
  score = attention_layer(transposed_x, transposed_mask)
  return tf.transpose(score, perm)

def AttentionLayer(num_blocks, num_heads, key_dim, inputs, inputs_mask):
  refactored_x = RefactorX(inputs, num_blocks)
  refactored_mask = RefactorXMask(tf.cast(inputs_mask, tf.bool), num_blocks)
  local_self_attention = CustomSelfAttention(num_heads, key_dim)(
      refactored_x, refactored_mask)
  global_self_attention = TransposeAndAttend(
      CustomSelfAttention(num_heads, key_dim), refactored_x,
      refactored_mask, [0, 2, 1, 3])
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

def TransformerLayer(num_transformers, num_blocks, num_heads, key_dim,
    num_dnn_layers, output_size, inputs, inputs_mask):
  x = inputs
  for i in range(num_transformers):
    x = AttentionLayer(num_blocks, num_heads, key_dim, x, inputs_mask)
    x = FeedForwardLayer(num_dnn_layers, output_size, x)
  return x

def _ApplySharedTransformer(base_features, atom_mask, num_blocks, num_transformer_channels):
  # Straighten, Attend, and Reshape
  original_shape = ShapeList(base_features)
  assert len(original_shape) == 4
  straightened_features = StraightenMultipeptideSequence(base_features)
  straightened_mask = StraightenMultipeptideMask(atom_mask)

  sequence_size = ShapeList(straightened_features)[1]
  ideal_sequence_size = tf.cast(
      tf.math.ceil(
        tf.cast(sequence_size, tf.float32)/num_blocks)*num_blocks, tf.int32)
  paddings = tf.stack([tf.constant([0, 0]),
                       tf.stack([tf.constant(0), ideal_sequence_size-sequence_size]),
                       tf.constant([0, 0])])
  padded_features = tf.pad(straightened_features, paddings)
  padded_features = tf.ensure_shape(padded_features,
      [None, None, num_transformer_channels])
  padded_mask = tf.pad(straightened_mask, tf.stack([
    tf.constant([0, 0]),
    tf.stack([tf.constant(0), ideal_sequence_size-sequence_size])]))
  transformer_output = TransformerLayer(10, num_blocks, 5, 10, 5,
      num_transformer_channels, padded_features, padded_mask)
  transformer_output = transformer_output[:,:sequence_size,:]
  transformer_output = tf.reshape(transformer_output,
      [original_shape[0], original_shape[1], original_shape[2], -1])
  return transformer_output

def log_sigma2(gamma):
  return tf.math.log_sigmoid(-1*gamma)

def sigma2(gamma):
  return tf.math.sigmoid(-1*gamma)

def alpha(gamma):
  return tf.math.sqrt(1-sigma2(gamma))

class ConstantValue(tf.keras.layers.Layer):
  def __init__(self, name, initial_value):
    super(ConstantValue, self).__init__()
    self._constant_value = tf.Variable(initial_value, name=name)

  def call(self, unused_input):
    return self._constant_value

def EncoderModel():
  # The inputs.
  normalized_coordinates = tf.keras.Input(
      shape=[None, None, 3],
      name='normalized_coordinates')

  gamma = ConstantValue('gamma', 6.0)(tf.constant(2.0))
  a = alpha(gamma)
  logvar = log_sigma2(gamma)
  return tf.keras.Model(
      inputs=[normalized_coordinates],
      outputs=tf.keras.layers.concatenate(inputs=
          [a*normalized_coordinates,
            logvar*tf.ones_like(normalized_coordinates)], axis=1))

def DecoderModel():
  # The inputs.
  z = tf.keras.Input(shape=[None, None, _LATENT_EMBEDDING_SIZE], name='z')
  atom_mask = tf.keras.Input(
      shape=[None, None],
      name='atom_mask')
  cond = tf.keras.Input(
      shape=[None, None, _COND_EMBEDDING_SIZE],
      name='cond')

  # Compute Amino Acid Positional Embedding
  pemb = AminoAcidPositionalEmbedding(cond)
  peptide_indx = PeptideIndx(cond)

  num_blocks = 200
  base_features = tf.keras.layers.concatenate(
      inputs=[z, cond, peptide_indx])
  transformer_output = _ApplySharedTransformer(
      base_features, atom_mask, num_blocks, 10)
  transformer_output = tf.ensure_shape(
      transformer_output, [None, None, None, 10])
  loc = tf.keras.layers.Dense(3)(transformer_output)
  scale_diag = ConstantValue('gamma', 1.0)(tf.constant(1.0))
  return tf.keras.Model(
      inputs=[z, atom_mask, cond],
      outputs=[loc, scale_diag*tf.ones_like(loc)])

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
  cond_out = tf.keras.layers.Dense(_COND_EMBEDDING_SIZE)(cond_out)
  return tf.keras.Model(inputs=[residue_names, atom_names], outputs=cond_out)

MODEL_FOR_TRAINING = lambda vocab: model.VariationalModel(
    model.Conditioner(
      CondModel(vocab.ResidueLookupSize(), vocab.AtomLookupSize())),
    model.Decoder(DecoderModel()),
    model.Encoder(EncoderModel()))
