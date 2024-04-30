import math
import tensorflow as tf
import random
import tf_keras

from protein_folding.variational_model import model

_COND_EMBEDDING_SIZE = 6
_LATENT_EMBEDDING_SIZE = 30
_AMINO_ACID_EMBEDDING_DIMS = 20
_NUM_TRANSFORMERS = 10
_ATOMS_PER_SEQUENCE = 6000
_BATCH_SIZE = 2
_NUM_PEPTIDES = 4
_CONFIG = None

_LOCAL_ATOMS_SIZE = 100

def AminoAcidPositionalEmbedding(cond):
  pos_indices = tf.expand_dims(
      tf.expand_dims(tf.expand_dims(
        tf.range(tf.shape(cond)[2], dtype='float32'), 0), 0) *
      tf.ones(tf.shape(cond)[:-1]), -1)
  half_dim = _AMINO_ACID_EMBEDDING_DIMS // 2
  pemb = tf.math.log(20_000.0) / (half_dim - 1)
  pemb = tf.math.exp(tf.range(half_dim, dtype='float') * - pemb)
  pemb = pos_indices * pemb
  pemb = tf_keras.layers.concatenate([tf.math.sin(pemb), tf.math.cos(pemb)])
  return pemb

def PeptideIndx():
  pos_indices = tf.range(_NUM_PEPTIDES, dtype='float32')
  pos_indices = tf.expand_dims(tf.expand_dims(tf.expand_dims(pos_indices, 0), -1), -1)
  pos_indices = pos_indices * tf.ones([_BATCH_SIZE, _NUM_PEPTIDES, _ATOMS_PER_SEQUENCE, 1])
  return pos_indices

def StraightenMultipeptideSequence(x, embedding_size):
  return tf.reshape(x,[
    _BATCH_SIZE, # Batch size remains the same.
    _NUM_PEPTIDES*_ATOMS_PER_SEQUENCE, # Amino acid dimension.
    embedding_size])

def StraightenMultipeptideMask(atom_mask):
  return tf.reshape(atom_mask, [
    _BATCH_SIZE, # Batch size remains the same.
    _NUM_PEPTIDES*_ATOMS_PER_SEQUENCE # Amino Acid dimension.
    ])

def RefactorX(x, timesteps, embedding_size, n):
  return tf.reshape(x, [
    _BATCH_SIZE, # Batch size remains the same.
    n, # Additional blocks are introduced.
    timesteps // n, # Number of timesteps reduced by a factor of n.
    embedding_size # Embedding dimension does not change.
    ])

def RefactorXMask(x_mask, timesteps,n):
  return tf.reshape(x_mask, [
    _BATCH_SIZE, # Batch size remains the same.
    n, # Additional blocks are introduced.
    timesteps // n # Number of timesteps reduced by a factor of n.
    ])

class CustomSelfAttention(tf_keras.layers.Layer):
  def __init__(self, num_heads, key_dim):
    super(CustomSelfAttention, self).__init__()
    self._attention_layer = tf_keras.layers.MultiHeadAttention(num_heads, key_dim)

  def call(self, input_tensor, input_mask):
    def _apply_attention(x):
      return self._attention_layer(x[0], x[0], x[0],
          tf.math.logical_and(
            tf.expand_dims(x[1], -1),
            tf.expand_dims(x[1], -2))
          )
    return tf.stack([
      _apply_attention((input_tensor[i], input_mask[i]))
      for i in range(_BATCH_SIZE)])

def TransposeAndAttend(attention_layer, refactored_x, refactored_mask, perm):
  transposed_x = tf.transpose(refactored_x, perm)
  transposed_mask = tf.transpose(refactored_mask, perm[:-1])
  score = attention_layer(transposed_x, transposed_mask)
  return tf.transpose(score, perm)

def AttentionLayer(num_blocks, num_heads, key_dim, timesteps, embedding_size, inputs, inputs_mask):
  refactored_x = RefactorX(inputs, timesteps, embedding_size, num_blocks)
  refactored_mask = RefactorXMask(tf.cast(inputs_mask, tf.bool), timesteps, num_blocks)
  local_self_attention = CustomSelfAttention(num_heads, key_dim)(
      refactored_x, refactored_mask)
  global_self_attention = TransposeAndAttend(
      CustomSelfAttention(num_heads, key_dim), refactored_x,
      refactored_mask, [0, 2, 1, 3])
  return tf_keras.layers.LayerNormalization(name='layer-normalization-1-'+str(random.randint(0, 10000)),
                                            axis=[1,2])(
      tf_keras.layers.Add()([
        inputs,
        tf.reshape(local_self_attention, [_BATCH_SIZE, timesteps, embedding_size]),
        tf.reshape(global_self_attention, [_BATCH_SIZE, timesteps, embedding_size])]))

def FeedForwardLayer(num_layers, ideal_sequence_size, output_size, inputs):
  t = inputs
  for i in range(num_layers):
    t = tf_keras.layers.Dense(100, 'gelu')(t)
  t = tf_keras.layers.Dense(output_size)(t)
  t = tf.ensure_shape(t, [_BATCH_SIZE, ideal_sequence_size, output_size])
  added_val = tf_keras.layers.Add()([inputs, t])
  added_val = tf.ensure_shape(added_val, [_BATCH_SIZE, ideal_sequence_size, output_size])
  return tf_keras.layers.LayerNormalization(name='layer-normalization-2-'+str(random.randint(0, 10000)),
                                            axis=[1,2])(
      added_val)

def EncoderTransformerLayer(num_transformers, num_blocks, num_heads, key_dim,
    num_dnn_layers, ideal_sequence_size, output_size, inputs, inputs_mask):
  x = inputs
  x_list = []
  for i in range(num_transformers):
    x = AttentionLayer(num_blocks, num_heads, key_dim, ideal_sequence_size, output_size, x, inputs_mask)
    x = tf.ensure_shape(x, [_BATCH_SIZE, ideal_sequence_size, output_size])
    x = FeedForwardLayer(num_dnn_layers, ideal_sequence_size, output_size, x)
    x_list.append(x)
  return x_list

def HierarchicalNoise(encoder_transformer_outputs, output_size, num_dense_layers):
  outputs = encoder_transformer_outputs
  for n in range(num_dense_layers):
    outputs = [
        tf_keras.layers.Dense(100, 'gelu')(e)
        for e in outputs]
  return [tf_keras.layers.Dense(output_size)(e) for e in outputs]

def _EncoderTransformer(base_features, atom_mask, num_blocks, num_transformer_channels):
  # Straighten, Attend, and Reshape
  straightened_features = StraightenMultipeptideSequence(base_features, num_transformer_channels)
  straightened_mask = StraightenMultipeptideMask(atom_mask)

  sequence_size = _NUM_PEPTIDES*_ATOMS_PER_SEQUENCE
  ideal_sequence_size = math.ceil(sequence_size/num_blocks) * num_blocks
  paddings = [[0, 0],
              [0, ideal_sequence_size-sequence_size],
              [0, 0]]
  padded_features = tf.pad(straightened_features, paddings)
  padded_features = tf.ensure_shape(padded_features,
      [_BATCH_SIZE, ideal_sequence_size, num_transformer_channels])
  padded_mask = tf.pad(straightened_mask, [
    [0, 0],
    [0, ideal_sequence_size-sequence_size]])
  transformer_outputs = EncoderTransformerLayer(
      num_transformers=_NUM_TRANSFORMERS,
      num_blocks=num_blocks,
      num_heads=5,
      key_dim=10,
      num_dnn_layers=5,
      ideal_sequence_size=ideal_sequence_size,
      output_size=num_transformer_channels,
      inputs=padded_features,
      inputs_mask=padded_mask)
  transformer_outputs = [t[:,:sequence_size,:] for t in transformer_outputs]
  transformer_outputs = HierarchicalNoise(
      transformer_outputs, num_transformer_channels, 2)

  transformer_outputs = [tf.reshape(t,
    [_BATCH_SIZE, _NUM_PEPTIDES, _ATOMS_PER_SEQUENCE, num_transformer_channels])
    for t in transformer_outputs]
  return transformer_outputs

def DecoderTransformLayer(num_blocks, num_heads, key_dim,
    num_dnn_layers, ideal_sequence_size, inputs, inputs_mask, z_list, channel_size):
  x = inputs
  # One layer per set of latent variables.
  num_layers = len(z_list)

  encoder_indices = range(num_layers-1, -1, -1)
  if _CONFIG.get('flip_encoder_processing', False):
    encoder_indices = range(num_layers)

  for i in encoder_indices:
    z = z_list[i]
    transformer_input = tf_keras.layers.concatenate(
        [x, z])
    transformer_input = tf.ensure_shape(transformer_input,
        [_BATCH_SIZE, ideal_sequence_size, channel_size + _LATENT_EMBEDDING_SIZE])
    transformer_input = AttentionLayer(
        num_blocks, num_heads, key_dim, ideal_sequence_size, channel_size + _LATENT_EMBEDDING_SIZE, transformer_input, inputs_mask)
    conv_inputs = tf_keras.layers.Conv1D(32, 100, padding='same')(
        transformer_input)
    conv_inputs = tf.ensure_shape(conv_inputs, [_BATCH_SIZE, ideal_sequence_size, 32])
    x = FeedForwardLayer(num_dnn_layers, ideal_sequence_size, channel_size,
        tf_keras.layers.Dense(channel_size)(
            tf_keras.layers.concatenate([transformer_input, conv_inputs])))
  return x

def _DecoderTransformer(
    base_features, z_list, atom_mask, num_blocks, channel_size):
  # Straighten, Attend, and Reshape
  straightened_features = StraightenMultipeptideSequence(base_features, channel_size)
  straightened_z_list = [StraightenMultipeptideSequence(z, _LATENT_EMBEDDING_SIZE) for z in z_list]
  straightened_mask = StraightenMultipeptideMask(atom_mask)

  # Pad the sequence to be evenly divisible by num_blocks
  sequence_size = _NUM_PEPTIDES*_ATOMS_PER_SEQUENCE
  ideal_sequence_size = math.ceil(sequence_size/num_blocks) * num_blocks
  paddings = [[0, 0],
              [0, ideal_sequence_size-sequence_size],
              [0, 0]]
  padded_features = tf.pad(straightened_features, paddings)
  padded_features = tf.ensure_shape(padded_features,
      [_BATCH_SIZE, ideal_sequence_size, channel_size])
  padded_z_list = [tf.pad(z, paddings) for z in straightened_z_list]
  padded_mask = tf.pad(straightened_mask, [
    [0, 0],
    [0, ideal_sequence_size-sequence_size]])
  transformer_outputs = DecoderTransformLayer(
      num_blocks=num_blocks,
      num_heads=5,
      key_dim=10,
      num_dnn_layers=5,
      ideal_sequence_size=ideal_sequence_size,
      inputs=padded_features,
      inputs_mask=padded_mask,
      z_list=padded_z_list,
      channel_size=channel_size)
  transformer_outputs = transformer_outputs[:,:sequence_size,:]
  transformer_outputs = tf.reshape(transformer_outputs,
      [_BATCH_SIZE, _NUM_PEPTIDES, _ATOMS_PER_SEQUENCE, channel_size])
  return transformer_outputs

def log_sigma2(gamma):
  return tf.math.log_sigmoid(-1*gamma)

def sigma2(gamma):
  return tf.math.sigmoid(-1*gamma)

def alpha(gamma):
  return tf.math.sqrt(1-sigma2(gamma))

class FinalEncoderLayer(tf_keras.layers.Layer):
  def __init__(self, initial_gamma_value):
    super(FinalEncoderLayer, self).__init__()
    self._initial_gamma_value = initial_gamma_value

  def build(self, input_shape):
    if _CONFIG.get('scale_coordinates', True):
      self._gamma_variable = tf.Variable(self._initial_gamma_value, name='gamma')

  def call(self, normalized_coordinates):
    a = 1.0
    logvar = 0
    if _CONFIG.get('scale_coordinates', True):
      a = alpha(self._gamma_variable)
      logvar = log_sigma2(self._gamma_variable)
    return tf_keras.layers.concatenate(
        inputs=[a*normalized_coordinates, logvar*tf.ones_like(normalized_coordinates)],
        axis=1)

def EncoderModel():
  # The inputs.
  normalized_coordinates = tf_keras.Input(
      shape=[_NUM_PEPTIDES, _ATOMS_PER_SEQUENCE, 3],
      name='normalized_coordinates')
  atom_mask = tf_keras.Input(
      shape=[_NUM_PEPTIDES, _ATOMS_PER_SEQUENCE],
      name='atom_mask')
  cond = tf_keras.Input(
      shape=[_NUM_PEPTIDES, _ATOMS_PER_SEQUENCE, _COND_EMBEDDING_SIZE],
      name='cond')

  # Compute Amino Acid Positional Embedding
  pemb = AminoAcidPositionalEmbedding(cond)
  peptide_indx = PeptideIndx()

  num_blocks = 200
  base_embedding_size = 3 + _COND_EMBEDDING_SIZE + _AMINO_ACID_EMBEDDING_DIMS + 1
  base_features = tf_keras.layers.concatenate(
      inputs=[normalized_coordinates, cond, pemb, peptide_indx])
  transformer_outputs = _EncoderTransformer(
      base_features, atom_mask, num_blocks, base_embedding_size)
  transformer_outputs = [
      tf.ensure_shape(t, [_BATCH_SIZE, _NUM_PEPTIDES, _ATOMS_PER_SEQUENCE, base_embedding_size])
      for t in transformer_outputs]
  transformer_outputs = tf_keras.layers.concatenate(
      [tf.expand_dims(t, 1) for t in transformer_outputs],
      axis=1)
  fel = FinalEncoderLayer(6.0)
  return tf_keras.Model(
      inputs=[normalized_coordinates, atom_mask, cond],
      outputs=fel(transformer_outputs))

class ClipMinMax(tf_keras.constraints.Constraint):
  def __init__(self, min_val, max_val):
      self._min_val = min_val
      self._max_val = max_val

  def __call__(self, var):
    return tf.clip_by_value(var, self._min_val, self._max_val)

class FinalDecoderLayer(tf_keras.layers.Layer):
  def __init__(self, initial_scale_diag_value):
    super(FinalDecoderLayer, self).__init__()
    self._initial_scale_diag_value = initial_scale_diag_value

  def build(self, input_shape):
    self._scale_diag_variable = tf.Variable(
            self._initial_scale_diag_value,
            name='scale_diag',
            constraint=ClipMinMax(0.001, 10))

  def call(self, loc):
    return [loc, self._scale_diag_variable*tf.ones_like(loc)]

class DecoderLSTM(tf_keras.layers.Layer):
  def __init__(self, units, batch_size):
    super(DecoderLSTM, self).__init__()
    self._lstm_layer = tf_keras.layers.Bidirectional(
        tf_keras.layers.LSTM(units, return_sequences=True))
    self._batch_size = batch_size
  
  def call(self, input_tensor, input_mask):
    return tf.stack([
      self._lstm_layer(inputs=input_tensor[i], mask=input_mask[i])
      for i in range(self._batch_size)])

def DecoderModel():
  # The inputs.
  z = tf_keras.Input(shape=[_NUM_TRANSFORMERS, _NUM_PEPTIDES, _ATOMS_PER_SEQUENCE, _LATENT_EMBEDDING_SIZE], name='z')
  # Drop some of the latent layer to improve robustness.
  dropout_z = tf_keras.layers.Dropout(0.2)(z)
  atom_mask = tf_keras.Input(
      shape=[_NUM_PEPTIDES, _ATOMS_PER_SEQUENCE],
      name='atom_mask')
  cond = tf_keras.Input(
      shape=[_NUM_PEPTIDES, _ATOMS_PER_SEQUENCE, _COND_EMBEDDING_SIZE],
      name='cond')

  z_list = tf.unstack(dropout_z, _NUM_TRANSFORMERS, axis=1)

  # Compute Amino Acid Positional Embedding
  pemb = AminoAcidPositionalEmbedding(cond)
  peptide_indx = PeptideIndx()

  num_blocks = 200
  base_embedding_size =_COND_EMBEDDING_SIZE + _AMINO_ACID_EMBEDDING_DIMS + 1
  base_features = tf_keras.layers.concatenate(
      inputs=[cond, pemb, peptide_indx])
  transformer_output = _DecoderTransformer(
      base_features=base_features,
      z_list=z_list,
      atom_mask=atom_mask,
      num_blocks=num_blocks,
      channel_size=base_embedding_size)
  transformer_output = tf.ensure_shape(
      transformer_output, [_BATCH_SIZE, _NUM_PEPTIDES, _ATOMS_PER_SEQUENCE, base_embedding_size])
  lstm_output = DecoderLSTM(64, _BATCH_SIZE)(transformer_output, tf.cast(atom_mask, tf.bool))
  loc = tf_keras.layers.Dense(3)(lstm_output)
  fdl = FinalDecoderLayer(1.0)
  return tf_keras.Model(
      inputs=[z, atom_mask, cond],
      outputs=fdl(loc))

def CondModel(residue_lookup_size, atom_lookup_size):
  residue_names = tf_keras.Input(shape=(_NUM_PEPTIDES, _ATOMS_PER_SEQUENCE), name='residue_names')
  atom_names = tf_keras.Input(shape=(_NUM_PEPTIDES, _ATOMS_PER_SEQUENCE), name='atom_names')

  residue_embeddings = tf_keras.layers.Embedding(
    input_dim=residue_lookup_size,
    output_dim=3)(residue_names)
  atom_embeddings = tf_keras.layers.Embedding(
    input_dim=atom_lookup_size,
    output_dim=3)(atom_names)

  cond_out = tf_keras.layers.concatenate(
    inputs=[residue_embeddings, atom_embeddings])
  cond_out = tf_keras.layers.Dense(100, 'gelu')(cond_out)
  cond_out = tf_keras.layers.Dense(_COND_EMBEDDING_SIZE)(cond_out)
  return tf_keras.Model(inputs=[residue_names, atom_names], outputs=cond_out)

def RotationModel():
  normalized_coordinates = tf_keras.Input(shape=[_NUM_PEPTIDES, _ATOMS_PER_SEQUENCE, 3],
                                          name='normalized_coordinates')
  atom_mask = tf_keras.Input(shape=[_NUM_PEPTIDES, _ATOMS_PER_SEQUENCE],
                             name='atom_mask')
  predicted_coordinates = tf_keras.Input(shape=[_NUM_PEPTIDES, _ATOMS_PER_SEQUENCE, 3],
                                         name='predicted_coordinates')
  input_features = tf_keras.layers.concatenate(
      [normalized_coordinates, predicted_coordinates])
  straightened_features = StraightenMultipeptideSequence(input_features, 6)
  straightened_mask = StraightenMultipeptideMask(atom_mask)
  straightened_features = (straightened_features *
                           tf.expand_dims(straightened_mask, -1))
  prediction = tf_keras.layers.Dense(3)(tf_keras.layers.Dense(100, 'gelu')(
      tf_keras.layers.Dense(100, 'gelu')(straightened_features)))
  prediction = prediction * tf.expand_dims(straightened_mask, -1)
  prediction = tf.math.reduce_sum(prediction, axis=1) / tf.expand_dims(tf.math.reduce_sum(straightened_mask, axis=1), -1)
  return tf_keras.Model(inputs=[normalized_coordinates, atom_mask, predicted_coordinates],
                        outputs=prediction)

def GlobalCoordinates(local_normalized_coordinates):
  local_normalized_coordinates = tf.ensure_shape(
      local_normalized_coordinates, [_BATCH_SIZE, _NUM_PEPTIDES, _ATOMS_PER_SEQUENCE//_LOCAL_ATOMS_SIZE, _LOCAL_ATOMS_SIZE, 3])
  return tf.reshape(local_normalized_coordinates, [_BATCH_SIZE, _NUM_PEPTIDES, _ATOMS_PER_SEQUENCE, 3])

def LocalCoordinates(normalized_coordinates):
  normalized_coordinates = tf.ensure_shape(
      normalized_coordinates, [_BATCH_SIZE, _NUM_PEPTIDES, _ATOMS_PER_SEQUENCE, 3])
  return tf.reshape(normalized_coordinates,
                    [_BATCH_SIZE,
                     _NUM_PEPTIDES,
                     _ATOMS_PER_SEQUENCE//_LOCAL_ATOMS_SIZE,
                     _LOCAL_ATOMS_SIZE, 3])

def LocalMask(atom_mask):
  atom_mask = tf.ensure_shape(
      atom_mask, [_BATCH_SIZE, _NUM_PEPTIDES, _ATOMS_PER_SEQUENCE])
  return tf.reshape(atom_mask,
                    [_BATCH_SIZE,
                     _NUM_PEPTIDES,
                     _ATOMS_PER_SEQUENCE//_LOCAL_ATOMS_SIZE,
                     _LOCAL_ATOMS_SIZE])

def LocalRotationModel():
  local_normalized_coordinates_mean_removed = tf_keras.Input(
      shape=[_NUM_PEPTIDES, _ATOMS_PER_SEQUENCE//_LOCAL_ATOMS_SIZE, _LOCAL_ATOMS_SIZE, 3],
      name='local_normalized_coordinates_mean_removed')
  local_atom_mask = tf_keras.Input(shape=[_NUM_PEPTIDES, _ATOMS_PER_SEQUENCE//_LOCAL_ATOMS_SIZE, _LOCAL_ATOMS_SIZE],
                                   name='local_atom_mask')
  num_local_atoms = tf_keras.Input(shape=[_NUM_PEPTIDES, _ATOMS_PER_SEQUENCE//_LOCAL_ATOMS_SIZE, 1],
                                   name='num_local_atoms')
  local_predicted_coordinates_mean_removed = tf_keras.Input(
      shape=[_NUM_PEPTIDES, _ATOMS_PER_SEQUENCE//_LOCAL_ATOMS_SIZE, _LOCAL_ATOMS_SIZE, 3],
      name='local_predicted_coordinates_mean_removed')

  input_features = tf_keras.layers.concatenate([
    local_normalized_coordinates_mean_removed,
    local_predicted_coordinates_mean_removed])

  prediction = tf_keras.layers.Dense(3)(tf_keras.layers.Dense(100, 'gelu')(
      tf_keras.layers.Dense(100, 'gelu')(input_features)))
  prediction = tf.math.divide_no_nan(tf.math.reduce_sum(
      prediction*tf.expand_dims(local_atom_mask, -1), -2), num_local_atoms)

  prediction = tf.ensure_shape(prediction, [_BATCH_SIZE, _NUM_PEPTIDES, _ATOMS_PER_SEQUENCE//_LOCAL_ATOMS_SIZE, 3])
  return tf_keras.Model(inputs=[local_normalized_coordinates_mean_removed, local_atom_mask, num_local_atoms, local_predicted_coordinates_mean_removed],
                        outputs=prediction)

def MODEL_FOR_TRAINING(vocab, config):
  global _CONFIG
  _CONFIG = config
  return model.VariationalModel(
      model.Conditioner(
        CondModel(vocab.ResidueLookupSize(), vocab.AtomLookupSize())),
      model.Decoder(DecoderModel()),
      model.Encoder(EncoderModel()),
      RotationModel(),
      model.LocalTransformationModel(LocalCoordinates, LocalMask, GlobalCoordinates) if config.get('should_do_local_transform', False) else None)
