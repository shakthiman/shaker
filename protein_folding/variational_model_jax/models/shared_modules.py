from flax import linen as nn
from jax import numpy as jnp

import typing

class DNN(nn.Module):
  dnn_layers: typing.List[nn.Dense]
  activation_function: typing.Callable[[typing.Any], typing.Any]

  def __call__(self, inputs):
    t = inputs
    for l in self.dnn_layers:
      t = l(t)
      t = self.activation_function(t)
    return t

class EfficientTransformerUnit(nn.Module):
  num_blocks: int
  input_length: int
  batch_size: int
  local_transformer: nn.MultiHeadAttention
  global_transformer: nn.MultiHeadAttention
  feedforward_network: DNN
  final_layer: nn.Dense
  dropout_fraction: typing.Optional[float] = None
  deterministic: bool = True

  def setup(self):
    assert self.input_length % self.num_blocks == 0
    self.attention_layer_norm = nn.LayerNorm(reduction_axes=[-2,-1],
                                             feature_axes=[-2,-1])
    self.feedforward_layer_norm = nn.LayerNorm(reduction_axes=[-2,-1],
                                               feature_axes=[-2,-1])
    if self.dropout_fraction:
      self.dropout_layer = nn.Dropout(self.dropout_fraction,
                                      deterministic=self.deterministic,
                                      broadcast_dims=(-1,))

  def _get_mask(self, mask):
    attention_mask = jnp.logical_and(jnp.expand_dims(mask, -1),
                                     jnp.expand_dims(mask, -2))
    return jnp.expand_dims(attention_mask, -3)

  def __call__(self, inputs, mask):
    reshaped_inputs = jnp.reshape(
        inputs, [self.batch_size, self.num_blocks,
                 self.input_length//self.num_blocks, -1])
    reshaped_mask = jnp.reshape(
        mask, [self.batch_size, self.num_blocks,
               self.input_length//self.num_blocks])
    local_self_attention = self.local_transformer(
        reshaped_inputs, mask=self._get_mask(reshaped_mask))
    global_self_attention = jnp.transpose(
        self.global_transformer(
            jnp.transpose(reshaped_inputs,axes=[0,2,1,3]),
            mask=self._get_mask(
                jnp.transpose(reshaped_mask,axes=[0,2,1]))),
        axes=[0,2,1,3])

    attention_values = (
        jnp.reshape(
          local_self_attention,
          [self.batch_size, self.input_length, -1])
        + jnp.reshape(
          global_self_attention,
          [self.batch_size, self.input_length, -1]))
    if self.dropout_fraction:
      attention_values = self.dropout_layer(
              attention_values)
    attention_values = (inputs + attention_values)
    attention_values = self.attention_layer_norm(attention_values)

    attention_values = self.feedforward_network(attention_values)
    attention_values = self.feedforward_layer_norm(attention_values)
    return self.final_layer(attention_values)

class EfficientTransformerUnit3(nn.Module):
  num_blocks: int
  input_length: int
  batch_size: int
  conv1: nn.Conv
  transformer2: nn.MultiHeadAttention
  transformer3: nn.MultiHeadAttention
  feedforward_network: DNN
  final_layer: nn.Dense
  dropout_fraction: typing.Optional[float] = None
  deterministic: bool = True

  def setup(self):
    assert self.input_length % self.num_blocks == 0
    self.attention_layer_norm = nn.LayerNorm(reduction_axes=[-2,-1],
                                             feature_axes=[-2,-1])
    self.feedforward_layer_norm = nn.LayerNorm(reduction_axes=[-2,-1],
                                               feature_axes=[-2,-1])
    if self.dropout_fraction:
      self.dropout_layer = nn.Dropout(self.dropout_fraction,
                                      deterministic=self.deterministic,
                                      broadcast_dims=(-1,))

  def _get_mask(self, mask):
    attention_mask = jnp.logical_and(jnp.expand_dims(mask, -1),
                                     jnp.expand_dims(mask, -2))
    return jnp.expand_dims(attention_mask, -3)

  def __call__(self, inputs, mask):
    reshaped_inputs = jnp.reshape(
        inputs, [self.batch_size, self.num_blocks, self.num_blocks,
                 self.input_length//self.num_blocks//self.num_blocks, -1])
    reshaped_mask = jnp.reshape(
        mask, [self.batch_size, self.num_blocks, self.num_blocks,
               self.input_length//self.num_blocks//self.num_blocks])
    attention1  = self.conv1(inputs)
    attention2 = jnp.transpose(self.transformer2(
      jnp.transpose(reshaped_inputs, axes=[0,1,3,2,4]),
      mask=self._get_mask(
        jnp.transpose(reshaped_mask, axes=[0,1,3,2]))),
      axes=[0,1,3,2,4])
    attention3 = jnp.transpose(self.transformer3(
      jnp.transpose(reshaped_inputs, axes=[0,3,2,1,4]),
      mask=self._get_mask(
        jnp.transpose(reshaped_mask, axes=[0,3,2,1]))),
      axes=[0,3,2,1,4])

    attention_values = (
        jnp.reshape(
          attention1,
          [self.batch_size, self.input_length, -1])
        + jnp.reshape(
          attention2,
          [self.batch_size, self.input_length, -1])
        + jnp.reshape(
          attention3,
          [self.batch_size, self.input_length, -1]))
    if self.dropout_fraction:
      attention_values = self.dropout_layer(
              attention_values)
    attention_values = (inputs + attention_values)
    attention_values = self.attention_layer_norm(attention_values)

    attention_values = self.feedforward_network(attention_values)
    attention_values = self.feedforward_layer_norm(attention_values)
    return self.final_layer(attention_values)
