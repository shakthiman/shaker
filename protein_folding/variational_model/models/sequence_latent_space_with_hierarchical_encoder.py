import tensorflow as tf

from protein_folding.variational_model import model

_COND_EMBEDDING_SIZE = 6
_LATENT_EMBEDDING_SIZE = 30
_AMINO_ACID_EMBEDDING_DIMS = 20
_NUM_TRANSFORMERS = 10

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

class TransposeAndAttend(tf.keras.layers.Layer):
  def __init__(self, num_heads, key_dim, value_dim,
               dotproduct_einsum_notation,
               kv_einsum_notation):
    super(TransposeAndAttend, self).__init__()
    self._num_heads = num_heads
    self._key_dim = key_dim
    self._value_dim = value_dim
    self._dotproduct_einsum_notation = dotproduct_einsum_notation
    self._kv_einsum_notation = kv_einsum_notation

  def build(self, input_shape):
    last_dim = input_shape[-1]

    self._query_projections = []
    self._key_projections = []
    self._value_projections = []
    self._dot_product_scales = []
    for i in range(self._num_heads):
      self._query_projections.append(
          self.add_weight(
            "query_projector",
            shape=[last_dim, self._key_dim],
            initializer="glorot_uniform",
            trainable=True))
      self._key_projections.append(
          self.add_weight(
            "key_projector",
            shape=[last_dim, self._key_dim],
            initializer="glorot_uniform",
            trainable=True))
      self._value_projections.append(
          self.add_weight(
            "value_projector",
            shape=[last_dim, self._value_dim],
            initializer="glorot_uniform",
            trainable=True))
      self._dot_product_scales.append(
          tf.Variable(1, name='dot_product_scale'))
    self._final_projection = self.add_weight(
        "final_projector",
        shape=[self._num_heads*self._value_dim, self._value_dim],
        initializer="glorot_uniform",
        trainable=True)

  def call(self, input_tensor, input_mask):
    intermediate_attentions = []
    for i in range(self._num_heads):
      qp = self._query_projections[i]
      kp = self._key_projections[i]
      vp = self._value_projections[i]
      ds = self._dot_product_scales[i]

      # Project (no transpose needed)
      qv = tf.linalg.matmul(input_tensor, qp)
      kv = tf.linalg.matmul(input_tensor, kp)
      vv = tf.linalg.matmul(input_tensor, vp)

      # Dot product (with transpose)
      attention_values = tf.einsum(self._dotproduct_einsum_notation, qv, kv)*ds
      orig_attention_values_shape = ShapeList(attention_values)
      qv_shape = ShapeList(qv)
      attention_values_long = tf.reshape(attention_values, qv_shape[:-1] + [-1])
      attention_values_long = tf.nn.softmax(attention_values_long)
      attention_values = tf.reshape(attention_values_long, orig_attention_values_shape)

      intermediate_attentions.append(
          tf.einsum(self._kv_einsum_notation, attention_values, vv,
            input_mask, input_mask))
    return tf.linalg.matmul(
        tf.keras.layers.concatenate(intermediate_attentions),
        self._final_projection)

def AttentionLayer(num_blocks, num_heads, key_dim, value_dim, inputs, inputs_mask):
  refactored_x = RefactorX(inputs, num_blocks)
  refactored_mask = RefactorXMask(inputs_mask, num_blocks)
  local_self_attention = TransposeAndAttend(num_heads, key_dim, value_dim,
                                            'bglc,bgkc->bglk',
                                            'bglk,bgkc,bgl,bgk->bglc')(
                                                refactored_x, refactored_mask)
  global_self_attention = TransposeAndAttend(num_heads, key_dim, value_dim,
                                             'bglc,bklc->bglk',
                                             'bglk,bklc,bgl,bkl->bglc')(refactored_x, refactored_mask)
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

def EncoderTransformerLayer(num_transformers, num_blocks, num_heads, key_dim,
    num_dnn_layers, output_size, inputs, inputs_mask):
  x = inputs
  x_list = []
  for i in range(num_transformers):
    x = AttentionLayer(num_blocks, num_heads, key_dim, output_size, x, inputs_mask)
    x = FeedForwardLayer(num_dnn_layers, output_size, x)
    x_list.append(x)
  return x_list

def HierarchicalNoise(encoder_transformer_outputs, output_size, num_dense_layers):
  outputs = encoder_transformer_outputs
  for n in range(num_dense_layers):
    outputs = [
        tf.keras.layers.Dense(100, 'gelu')(e)
        for e in outputs]
  return [tf.keras.layers.Dense(output_size)(e) for e in outputs]

def _EncoderTransformer(base_features, atom_mask, num_blocks, num_transformer_channels):
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
  transformer_outputs = EncoderTransformerLayer(
      num_transformers=_NUM_TRANSFORMERS,
      num_blocks=num_blocks,
      num_heads=5,
      key_dim=10,
      num_dnn_layers=5,
      output_size=num_transformer_channels,
      inputs=padded_features,
      inputs_mask=padded_mask)
  transformer_outputs = [t[:,:sequence_size,:] for t in transformer_outputs]
  transformer_outputs = HierarchicalNoise(
      transformer_outputs, num_transformer_channels, 2)

  transformer_outputs = [tf.reshape(t,
    [original_shape[0], original_shape[1], original_shape[2], -1])
    for t in transformer_outputs]
  return transformer_outputs

def DecoderTransformLayer(num_blocks, num_heads, key_dim,
    num_dnn_layers, inputs, inputs_mask, z_list, channel_size):
  x = inputs
  # One layer per set of latent variables.
  num_layers = len(z_list)

  for i in range(num_layers-1, -1, -1):
    z = z_list[i]
    transformer_input = tf.keras.layers.concatenate(
        [x, z])
    transformer_input = tf.ensure_shape(transformer_input,
        [None, None, channel_size + _LATENT_EMBEDDING_SIZE])
    transformer_input = AttentionLayer(
        num_blocks, num_heads, key_dim, channel_size + _LATENT_EMBEDDING_SIZE,
        transformer_input, inputs_mask)
    conv_inputs = tf.keras.layers.Conv1D(32, 100, padding='same')(transformer_input)
    x = FeedForwardLayer(num_dnn_layers, channel_size,
        tf.keras.layers.Dense(channel_size)(
            tf.keras.layers.concatenate([transformer_input, conv_inputs])))
  return x

def _DecoderTransformer(
    base_features, z_list, atom_mask, num_blocks, channel_size):
  # Straighten, Attend, and Reshape
  original_shape = ShapeList(base_features)
  assert len(original_shape) == 4
  straightened_features = StraightenMultipeptideSequence(base_features)
  straightened_z_list = [StraightenMultipeptideSequence(z) for z in z_list]
  straightened_mask = StraightenMultipeptideMask(atom_mask)

  # Pad the sequence to be evenly divisible by num_blocks
  sequence_size = ShapeList(straightened_features)[1]
  ideal_sequence_size = tf.cast(
      tf.math.ceil(
        tf.cast(sequence_size, tf.float32)/num_blocks)*num_blocks, tf.int32)
  paddings = tf.stack([tf.constant([0, 0]),
                       tf.stack([tf.constant(0), ideal_sequence_size-sequence_size]),
                       tf.constant([0, 0])])
  padded_features = tf.pad(straightened_features, paddings)
  padded_features = tf.ensure_shape(padded_features,
      [None, None, channel_size])
  padded_z_list = [tf.pad(z, paddings) for z in straightened_z_list]
  padded_mask = tf.pad(straightened_mask, tf.stack([
    tf.constant([0, 0]),
    tf.stack([tf.constant(0), ideal_sequence_size-sequence_size])]))
  transformer_outputs = DecoderTransformLayer(
      num_blocks=num_blocks,
      num_heads=5,
      key_dim=10,
      num_dnn_layers=5,
      inputs=padded_features,
      inputs_mask=padded_mask,
      z_list=padded_z_list,
      channel_size=channel_size)
  transformer_outputs = transformer_outputs[:,:sequence_size,:]
  transformer_outputs = tf.reshape(transformer_outputs,
      [original_shape[0], original_shape[1], original_shape[2], -1])
  return transformer_outputs

def log_sigma2(gamma):
  return tf.math.log_sigmoid(-1*gamma)

def sigma2(gamma):
  return tf.math.sigmoid(-1*gamma)

def alpha(gamma):
  return tf.math.sqrt(1-sigma2(gamma))

class FinalEncoderLayer(tf.keras.layers.Layer):
  def __init__(self, initial_gamma_value):
    super(FinalEncoderLayer, self).__init__()
    self._initial_gamma_value = initial_gamma_value

  def build(self, input_shape):
    self._gamma_variable = tf.Variable(self._initial_gamma_value, name='gamma')

  def call(self, normalized_coordinates):
    a = alpha(self._gamma_variable)
    logvar = log_sigma2(self._gamma_variable)
    return tf.keras.layers.concatenate(
        inputs=[a*normalized_coordinates, logvar*tf.ones_like(normalized_coordinates)],
        axis=1)

def EncoderModel():
  # The inputs.
  normalized_coordinates = tf.keras.Input(
      shape=[None, None, 3],
      name='normalized_coordinates')
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
      inputs=[normalized_coordinates, cond, pemb, peptide_indx])
  transformer_outputs = _EncoderTransformer(
      base_features, atom_mask, num_blocks, 30)
  transformer_outputs = [
      tf.ensure_shape(t, [None, None, None, 30])
      for t in transformer_outputs]
  transformer_outputs = tf.keras.layers.concatenate(
      [tf.expand_dims(t, 1) for t in transformer_outputs],
      axis=1)
  fel = FinalEncoderLayer(6.0)
  return tf.keras.Model(
      inputs=[normalized_coordinates, atom_mask, cond],
      outputs=fel(transformer_outputs))

class ClipMinMax(tf.keras.constraints.Constraint):
  def __init__(self, min_val, max_val):
      self._min_val = min_val
      self._max_val = max_val

  def __call__(self, var):
    return tf.clip_by_value(var, self._min_val, self._max_val)

class FinalDecoderLayer(tf.keras.layers.Layer):
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

def DecoderModel():
  # The inputs.
  z = tf.keras.Input(shape=[None, None, None, _LATENT_EMBEDDING_SIZE], name='z')
  # Drop some of the latent layer to improve robustness.
  dropout_z = tf.keras.layers.Dropout(0.2)(z)
  atom_mask = tf.keras.Input(
      shape=[None, None],
      name='atom_mask')
  cond = tf.keras.Input(
      shape=[None, None, _COND_EMBEDDING_SIZE],
      name='cond')

  z_list = tf.unstack(dropout_z, _NUM_TRANSFORMERS, axis=1)

  # Compute Amino Acid Positional Embedding
  pemb = AminoAcidPositionalEmbedding(cond)
  peptide_indx = PeptideIndx(cond)

  num_blocks = 200
  base_features = tf.keras.layers.concatenate(
      inputs=[cond, pemb, peptide_indx])
  transformer_output = _DecoderTransformer(
      base_features=base_features,
      z_list=z_list,
      atom_mask=atom_mask,
      num_blocks=num_blocks,
      channel_size=27)
  transformer_output = tf.ensure_shape(
      transformer_output, [None, None, None, 27])
  loc = tf.keras.layers.Dense(3)(transformer_output)
  fdl = FinalDecoderLayer(1.0)
  return tf.keras.Model(
      inputs=[z, atom_mask, cond],
      outputs=fdl(loc))

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
