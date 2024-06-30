import collections

import jax
from jax import numpy as jnp
from jax import scipy as jsp

LossParams = collections.namedtuple(
    'LossParams', ['batch_size', 'input_length', 'alpha_carbon'])
ClashParams = collections.namedtuple(
    'ClashParams', ['nearby_size'])

ClashLoss = collections.namedtuple(
    'ClashLoss', ['num_hard_clashes', 'num_soft_clashes'])
def Clashes(mask, normalized_coordinates, training_data,
            loss_params, clash_params):
  def _SingleBatchLoss(single_batch_data):
    s_mask = single_batch_data['mask']
    s_normalized_coordinates = single_batch_data['normalized_coordinates']
    s_is_alpha_carbon = single_batch_data['is_alpha_carbon']

    def _1DShift(arr, num):
      shifted_array = jnp.pad(arr, [(0, num)])
      return shifted_array[num:]
    def _2DShift(arr, num):
      shifted_array = jnp.pad(arr, [(0, num),
                                    (0, 0)])
      return shifted_array[num:,:]

    neighborhood_mask = jnp.concatenate([
        jnp.expand_dims(_1DShift(s_mask, num), axis=1)
        for num in range(1,clash_params.nearby_size)], axis=1)

    neighborhood_normalized_coordinates = jnp.concatenate([
        jnp.expand_dims(_2DShift(s_normalized_coordinates, num), axis=1)
        for num in range(1,clash_params.nearby_size)], axis=1)
    neighborhood_is_alpha_carbon = jnp.concatenate([
        jnp.expand_dims(_1DShift(s_is_alpha_carbon, num), axis=1)
        for num in range(1,clash_params.nearby_size)], axis=1)

    l2_distances = jnp.sum(jnp.square(
        neighborhood_normalized_coordinates -
        jnp.expand_dims(s_normalized_coordinates, axis=1)),
                             axis=-1)
    is_hard_clash = jnp.less_equal(l2_distances, jnp.square(3.5))
    is_soft_clash = jnp.less_equal(l2_distances, jnp.square(3.6))
    num_hard_clashes= jnp.sum(jnp.where(
      jnp.logical_and(
        jnp.logical_and(s_mask,
                        s_is_alpha_carbon),
        jnp.logical_and(neighborhood_mask,
                        neighborhood_is_alpha_carbon)),
        is_hard_clash, 0), axis=[0,1])
    num_soft_clashes= jnp.sum(jnp.where(
      jnp.logical_and(
        jnp.logical_and(
          jnp.logical_and(s_mask,
                          s_is_alpha_carbon),
          jnp.logical_and(neighborhood_mask,
                          neighborhood_is_alpha_carbon)),
          is_soft_clash),
      jax.nn.sigmoid(20*(jnp.square(3.55)-l2_distances)), 0), axis=[0,1])
    return (num_hard_clashes, num_soft_clashes)
  num_hard_clashes, num_soft_clashes = jax.lax.map(
      _SingleBatchLoss, {'mask': mask,
                         'normalized_coordinates': normalized_coordinates,
                         'is_alpha_carbon': jnp.equal(training_data['atom_names'],
                                                      loss_params.alpha_carbon)})
  return ClashLoss(num_hard_clashes=jnp.mean(num_hard_clashes, axis=0),
                   num_soft_clashes=jnp.mean(num_soft_clashes, axis=0))
