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
    def _SingleAtomLoss(atom_idx):
      s_mask = single_batch_data['mask']
      s_normalized_coordinates = single_batch_data['normalized_coordinates']
      s_is_alpha_carbon = single_batch_data['is_alpha_carbon']

      first_idx = min(loss_params.input_length-1,atom_idx+1)
      last_idx = min(loss_params.input_length-1,
                     atom_idx+clash_params.nearby_size+1)
      neighborhood_mask = s_mask[first_idx:last_idx]
      neighborhood_normalized_coordinates = s_normalized_coordinates[
          first_idx:last_idx,:]
      neighborhood_is_alpha_carbon = s_is_alpha_carbon[
          first_idx:last_idx]

      my_coordinate = s_normalized_coordinates[atom_idx,:]
      l2_distances = jnp.sum(jnp.square(
        neighborhood_normalized_coordinates - my_coordinate),
                             axis=-1)
      is_hard_clash = jnp.less_equal(l2_distances, jnp.square(3.5))
      is_soft_clash = jnp.less_equal(l2_distances, jnp.square(3.6))
      num_hard_clashes= jnp.sum(jnp.where(
          jnp.logical_and(
            jnp.logical_and(s_mask[atom_idx],
                            s_is_alpha_carbon[atom_idx]),
            jnp.logical_and(neighborhood_mask,
                           neighborhood_is_alpha_carbon)),
          is_hard_clash, 0), axis=0)
      num_soft_clashes= jnp.sum(jnp.where(
          jnp.logical_and(
            jnp.logical_and(
              jnp.logical_and(s_mask[atom_idx],
                              s_is_alpha_carbon[atom_idx]),
              jnp.logical_and(neighborhood_mask,
                              neighborhood_is_alpha_carbon)),
              is_soft_clash),
          jax.nn.sigmoid(20*(jnp.square(3.55)-l2_distances)), 0), axis=0)
      return (num_hard_clashes, num_soft_clashes)
    total_loss = (0, 0)
    for i in range(loss_params.input_length-1):
      sl = _SingleAtomLoss(i)
      total_loss[0] = total_loss[0] + sl[0]
      total_loss[1] = total_loss[1] + sl[1]
    return total_loss
  num_hard_clashes, num_soft_clashes = jax.lax.map(
      _SingleBatchLoss, {'mask': mask,
                         'normalized_coordinates': normalized_coordinates,
                         'is_alpha_carbon': jnp.equal(training_data['atom_names'],
                                                      loss_params.alpha_carbon)})
  return ClashLoss(num_hard_clashes=jnp.mean(num_hard_clashes, axis=0),
                   num_soft_clashes=jnp.mean(num_soft_clashes, axis=0))
