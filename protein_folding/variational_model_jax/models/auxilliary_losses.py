import collections

import jax
from jax import numpy as jnp

LossParams = collections.namedtuple(
    'LossParams', ['batch_size', 'input_length', 'alpha_carbon'])

ClashLoss = collections.namedtuple(
    'ClashLoss', ['num_hard_clashes', 'num_soft_clashes'])
def Clashes(mask, normalized_coordinates, training_data,
            loss_params):
  def _SingleBatchLoss(single_batch_data):
    compression = jnp.logical_and(single_batch_data['mask'],
                                  single_batch_data['is_alpha_carbon'])
    relevant_coordinates = jnp.compress(
        compression, single_batch_data['normalized_coordinates'], axis=0)
    l2_distances = jnp.sum(
        jnp.square(jnp.expand_dims(relevant_coordinates, -2) -
                   jnp.expand_dims(relevant_coordinates, -3)),
        axis=-1)

    not_same_atom = jnp.arange(0, loss_params.input_length) 
    not_same_atom = jnp.compress(compression, not_same_atom, axis=0)
    not_same_atom = jnp.not_equal(jnp.expand_dims(not_same_atom, -1),
                                  jnp.expand_dims(not_same_atom, -2))

    is_hard_clash = jnp.less_equal(l2_distances, jnp.square(3.5))
    num_hard_clashes = jnp.sum(
        is_hard_clash * not_same_atom,
        axis=(0, 1))

    is_soft_clash = jnp.less_equal(l2_distances, jnp.square(3.6))
    num_soft_clashes = jnp.sum(jax.nn.sigmoid(20*(jnp.square(3.55)-l2_distances))
                               * not_same_atom * is_soft_clash,
                               axis=(0,1))
    return (num_hard_clashes, num_soft_clashes)

  num_hard_clashes, num_soft_clashes = jax.lax.map(
      _SingleBatchLoss, {'mask': mask,
                         'normalized_coordinates': normalized_coordinates,
                         'is_alpha_carbon': jnp.equal(training_data['atom_names'],
                                                      loss_params.alpha_carbon)})
  return ClashLoss(num_hard_clashes=jnp.mean(num_hard_clashes, axis=0),
                   num_soft_clashes=jnp.mean(num_soft_clashes, axis=0))
