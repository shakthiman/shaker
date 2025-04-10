import collections

import jax
from jax import numpy as jnp
from jax import scipy as jsp

LossParams = collections.namedtuple(
    'LossParams', ['batch_size', 'input_length', 'alpha_carbon'])
ClashParams = collections.namedtuple(
    'ClashParams', ['nearby_size'])

ClashLoss = collections.namedtuple(
    'ClashLoss', ['num_hard_clashes', 'num_soft_clashes', 'sum_squares'])

def _1DShift(arr, num):
  shifted_array = jnp.pad(arr, [(0, num)])
  return shifted_array[num:]
def _2DShift(arr, num):
  shifted_array = jnp.pad(arr, [(0, num),
                                (0, 0)])
  return shifted_array[num:,:]

def Clashes(mask, normalized_coordinates, training_data,
            loss_params, clash_params):
  def _SingleBatchLoss(single_batch_data):
    s_mask = single_batch_data['mask']
    s_normalized_coordinates = single_batch_data['normalized_coordinates']
    s_is_alpha_carbon = single_batch_data['is_alpha_carbon']

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
        jnp.expand_dims(jnp.logical_and(
          s_mask, s_is_alpha_carbon), axis=1),
        jnp.logical_and(neighborhood_mask,
                        neighborhood_is_alpha_carbon)),
        is_hard_clash, 0), axis=[0,1])

    soft_clash_condition = jnp.logical_and(
        jnp.logical_and(
          jnp.expand_dims(jnp.logical_and(
            s_mask, s_is_alpha_carbon), axis=1),
          jnp.logical_and(neighborhood_mask,
                          neighborhood_is_alpha_carbon)),
          is_soft_clash)
    num_soft_clashes= jnp.sum(jnp.where(
      soft_clash_condition,
      jax.nn.sigmoid(20*(jnp.square(3.55)-l2_distances)), 0), axis=[0,1])
    sum_squares = jnp.sum(jnp.where(
      soft_clash_condition,
      - l2_distances + jnp.square(3.6), 0), axis=[0,1])
    return (num_hard_clashes, num_soft_clashes, sum_squares)
  num_hard_clashes, num_soft_clashes, sum_squares = jax.lax.map(
      _SingleBatchLoss, {'mask': mask,
                         'normalized_coordinates': normalized_coordinates,
                         'is_alpha_carbon': jnp.equal(training_data['atom_names'],
                                                      loss_params.alpha_carbon)})
  return ClashLoss(num_hard_clashes=jnp.mean(num_hard_clashes, axis=0),
                   num_soft_clashes=jnp.mean(num_soft_clashes, axis=0),
                   sum_squares=jnp.mean(sum_squares, axis=0))

DihedralParams = collections.namedtuple(
    'DihedralParams', ['carbon', 'nitrogen', 'nearby_size'])
DihedralLoss = collections.namedtuple(
    'DihedralLoss', ['total_phi_error', 'total_psi_error', 'total_omega_error',
                     'average_phi_error', 'average_psi_error', 'average_omega_error'])
def DihedralLosses(mask, predicted_coordinates, training_data, loss_params,
                   dihedral_params):
  def _SingleBatchLoss(single_batch_data):
    s_mask = single_batch_data['mask']
    s_true_normalized_coordinates = single_batch_data['true_normalized_coordinates']
    s_predicted_coordinates = single_batch_data['predicted_coordinates']
    s_peptide_idx = single_batch_data['peptide_idx']
    s_is_alpha_carbon = single_batch_data['is_alpha_carbon']
    s_is_carbon = single_batch_data['is_carbon']
    s_is_nitrogen = single_batch_data['is_nitrogen']

    next_is_alpha_carbon = [
        _1DShift(s_is_alpha_carbon, num)
        for num in range(1, dihedral_params.nearby_size)]
    next_is_carbon = [
        _1DShift(s_is_carbon, num)
        for num in range(1, dihedral_params.nearby_size)]
    next_is_nitrogen = [
        _1DShift(s_is_nitrogen, num)
        for num in range(1, dihedral_params.nearby_size)]
    next_peptide_idx = [
        _1DShift(s_peptide_idx, num)
        for num in range(1, dihedral_params.nearby_size)]


    next_atom_match = [
          jnp.logical_and(
            jnp.equal(s_peptide_idx, next_peptide_idx[i]),
            jnp.logical_or(
              jnp.logical_or(
                jnp.logical_and(s_is_alpha_carbon, next_is_carbon[i]),
                jnp.logical_and(s_is_carbon, next_is_nitrogen[i])),
              jnp.logical_and(s_is_nitrogen, next_is_alpha_carbon[i])))
            for i in range(len(next_is_alpha_carbon))]
    # Check if there was a match.
    matched = jnp.any(jnp.stack(next_atom_match), axis=0)
    next_matched = [
        _1DShift(matched, num)
        for num in range(1, dihedral_params.nearby_size)]
    u2_match = [
        jnp.logical_and(
          jnp.logical_and(next_matched[i],
                          jnp.equal(s_peptide_idx, next_peptide_idx[i])),
        jnp.logical_or(
          jnp.logical_or(
            jnp.logical_and(s_is_alpha_carbon, next_is_carbon[i]),
            jnp.logical_and(s_is_carbon, next_is_nitrogen[i])),
          jnp.logical_and(s_is_nitrogen, next_is_alpha_carbon[i])))
          for i in range(len(next_is_carbon))]
    u3_match = [
        jnp.logical_and(
          jnp.logical_and(next_matched[i],
                          jnp.equal(s_peptide_idx, next_peptide_idx[i])),
        jnp.logical_or(
          jnp.logical_or(
            jnp.logical_and(s_is_alpha_carbon, next_is_nitrogen[i]),
            jnp.logical_and(s_is_carbon, next_is_alpha_carbon[i])),
          jnp.logical_and(s_is_nitrogen, next_is_carbon[i])))
          for i in range(len(next_is_carbon))]
    u2_matched = jnp.any(jnp.stack(u2_match), axis=0)
    u3_matched = jnp.any(jnp.stack(u3_match), axis=0)

    def _CrossProduct(a, b):
      a1 = a[:,0]
      a2 = a[:,1]
      a3 = a[:,2]

      b1 = b[:,0]
      b2 = b[:,1]
      b3 = b[:,2]

      return jnp.stack([
        a2*b3 - a3*b2,
        a3*b1 - a1*b3,
        a1*b2 - a2*b1], axis=1)

    def _ComputeAngle(coordinates):
      next_coordinates = [_2DShift(coordinates, num)
                          for num in range(1, dihedral_params.nearby_size)] 
      next_coordinate = jnp.select([jnp.expand_dims(a, 1)
                                    for a in next_atom_match], next_coordinates)
      u1 = next_coordinate - coordinates
      # Shift the u_s to find u_2 and u_3
      next_us = [
          _2DShift(u1, num)
          for num in range(1, dihedral_params.nearby_size)]
      u2 = jnp.select([jnp.expand_dims(a, 1)
                       for a in u2_match], next_us)
      u3 = jnp.select([jnp.expand_dims(a, 1)
                       for a in u3_match], next_us)
      u1_u2_cross = _CrossProduct(u1, u2)
      u2_u3_cross = _CrossProduct(u2, u3)
      l = jnp.sum(
          u2*(_CrossProduct(u1_u2_cross, u2_u3_cross)),
          axis=1)
      r = jnp.linalg.norm(u2, axis=1)*jnp.sum(
          u1_u2_cross*u2_u3_cross, axis=1)
      l = jnp.where(jnp.absolute(l)<1e-6, jnp.ones_like(l), l)
      r = jnp.where(jnp.absolute(r)<1e-6, jnp.ones_like(r), r)
      angle = jnp.arctan2(l, r)
      return angle

    angle_true = _ComputeAngle(s_true_normalized_coordinates)
    angle_predicted = _ComputeAngle(s_predicted_coordinates)

    current_atom_is_eligible = jnp.logical_and(
        jnp.logical_and(matched, u2_matched), u3_matched)

    current_atom_is_phi = jnp.logical_and(
        current_atom_is_eligible, s_is_carbon)
    current_atom_is_psi = jnp.logical_and(
        current_atom_is_eligible, s_is_nitrogen)
    current_atom_is_omega = jnp.logical_and(
        current_atom_is_eligible, s_is_alpha_carbon)

    angle_error = jnp.mod(angle_true-angle_predicted + 2*jnp.pi, 2*jnp.pi)
    angle_error_rev  = jnp.mod(angle_predicted-angle_true + 2*jnp.pi, 2*jnp.pi)
    angle_error = jnp.minimum(angle_error, angle_error_rev)
    total_phi_error = jnp.sum(
        jnp.where(current_atom_is_phi, angle_error, 0),
        axis=0)
    total_psi_error = jnp.sum(
        jnp.where(current_atom_is_psi, angle_error, 0),
        axis=0)
    total_omega_error = jnp.sum(
        jnp.where(current_atom_is_omega, angle_error, 0),
        axis=0)

    average_phi_error = total_phi_error / jnp.sum(current_atom_is_phi, axis=0)
    average_psi_error = total_psi_error / jnp.sum(current_atom_is_psi, axis=0)
    average_omega_error = total_omega_error / jnp.sum(current_atom_is_omega, axis=0)
    return (total_phi_error, total_psi_error, total_omega_error,
            average_phi_error, average_psi_error, average_omega_error)

  (total_phi_error, total_psi_error, total_omega_error,
   average_phi_error, average_psi_error, average_omega_error) = jax.lax.map(
       _SingleBatchLoss, {
         'mask': mask,
         'true_normalized_coordinates': training_data['normalized_coordinates'],
         'predicted_coordinates': predicted_coordinates,
         'peptide_idx': training_data['peptide_indices'],
         'is_alpha_carbon': jnp.equal(training_data['atom_names'],
                                      loss_params.alpha_carbon),
         'is_carbon': jnp.equal(training_data['atom_names'],
                                dihedral_params.carbon),
         'is_nitrogen': jnp.equal(training_data['atom_names'],
                                dihedral_params.nitrogen)})
  return DihedralLoss(
      total_phi_error=jnp.mean(total_phi_error, axis=0),
      total_psi_error=jnp.mean(total_psi_error, axis=0),
      total_omega_error=jnp.mean(total_omega_error, axis=0),
      average_phi_error=jnp.mean(average_phi_error, axis=0),
      average_psi_error=jnp.mean(average_psi_error, axis=0),
      average_omega_error=jnp.mean(average_omega_error, axis=0))

RadiusOfGyrationLoss = collections.namedtuple(
    'RadiusOfGyrationLoss', ['alpha_carbon_only_radius_of_gyration_diff',
                             'alpha_carbon_only_radius_of_gyration_squared_diff'])
def RadiusOfGyration(mask, predicted_coordinates, training_data, loss_params):
  def _RadiusOfGyrationSquared(coordinates):
    is_alpha_carbon = jnp.equal(training_data['atom_names'],
                                loss_params.alpha_carbon)
    should_consider_atom = jnp.logical_and(mask, is_alpha_carbon)
    sum_coordinates = jnp.sum(
        jnp.where(
          jnp.expand_dims(should_consider_atom, -1),
          coordinates, 0), axis=1)
    num_alpha_carbons = jnp.sum(
        jnp.where(should_consider_atom, 1, 0), axis=1)
    mean_coordinate = sum_coordinates / num_alpha_carbons

    l2_distance_from_centroid = jnp.sum(
        jnp.square(coordinates
                   - jnp.expand_dims(mean_coordinate, axis=1)), axis=-1)
    radius_of_gyration_squared = jnp.sum(
        jnp.where(is_alpha_carbon, l2_distance_from_centroid, 0), axis=1) / num_alpha_carbons
    return radius_of_gyration_squared

  true_radius_of_gyration_squared = _RadiusOfGyrationSquared(training_data['normalized_coordinates'])
  predicted_radius_of_gyration_squared = _RadiusOfGyrationSquared(predicted_coordinates)
  diff_radius_of_gyration_squared = jnp.abs(
      true_radius_of_gyration_squared
      - predicted_radius_of_gyration_squared)
  diff_radius_of_gyration = jnp.abs(
      jnp.sqrt(true_radius_of_gyration_squared)
      - jnp.sqrt(predicted_radius_of_gyration_squared))
  return RadiusOfGyrationLoss(
      alpha_carbon_only_radius_of_gyration_diff=jnp.mean(diff_radius_of_gyration, axis=0),
      alpha_carbon_only_radius_of_gyration_squared_diff=jnp.mean(diff_radius_of_gyration_squared, axis=0))

DistanceMatrixParams = collections.namedtuple(
    'DistanceMatrixParams', ['nearby_size'])
DistanceMatrixLoss = collections.namedtuple(
    'DistanceMatrixLoss', ['alpha_carbon_squared_distances_loss',
                           'alpha_carbon_avg_distance_error'])
def DistanceMatrix(mask, predicted_coordinates, training_data, loss_params,
                   distance_matrix_params):
  def _DistanceMatrix(s_coordinates):
    neighborhood_coordinates = jnp.concatenate([
      jnp.expand_dims(_2DShift(s_coordinates, num), axis=1)
      for num in range(1, distance_matrix_params.nearby_size)], axis=1)
    l2_distances = jnp.sum(jnp.square(
      neighborhood_coordinates -
      jnp.expand_dims(s_coordinates, axis=1)), axis=-1)
    return l2_distances

  def _SingleBatchLoss(single_batch_data):
    s_mask = single_batch_data['mask']
    s_normalized_coordinates = single_batch_data['normalized_coordinates']
    s_predicted_coordinates = single_batch_data['predicted_coordinates']
    s_is_alpha_carbon = single_batch_data['is_alpha_carbon']

    should_consider_atom = jnp.logical_and(s_mask, s_is_alpha_carbon)
    should_consider_neighbor = jnp.concatenate([
      jnp.expand_dims(_1DShift(should_consider_atom, num), axis=1)
      for num in range(1, distance_matrix_params.nearby_size)], axis=1)
    should_consider_entry = jnp.logical_and(
        jnp.expand_dims(should_consider_atom, axis=1),
        should_consider_neighbor)

    true_distance_matrix = _DistanceMatrix(s_normalized_coordinates)
    predicted_distance_matrix = _DistanceMatrix(s_predicted_coordinates)

    squared_distance_loss = jnp.sum(
        jnp.where(
          should_consider_entry,
          jnp.abs(true_distance_matrix - predicted_distance_matrix),
          0),
        axis=[0,1])
    distance_error = jnp.sum(
        jnp.where(
          should_consider_entry,
          jnp.abs(
            jnp.sqrt(true_distance_matrix)
            - jnp.sqrt(predicted_distance_matrix)),
          0),
        axis=[0,1])
    total_entries_considered = jnp.sum(
        should_consider_entry, axis=[0,1])
    avg_distance_error = distance_error/total_entries_considered
    return (squared_distance_loss, avg_distance_error)
  (squared_distance_loss, avg_distance_error) = jax.lax.map(
      _SingleBatchLoss, {
        'mask': mask,
        'normalized_coordinates': training_data['normalized_coordinates'],
        'predicted_coordinates': predicted_coordinates,
        'is_alpha_carbon': jnp.equal(training_data['atom_names'],
                                     loss_params.alpha_carbon)})
  return DistanceMatrixLoss(
          alpha_carbon_squared_distances_loss=jnp.mean(squared_distance_loss, axis=0),
          alpha_carbon_avg_distance_error=jnp.mean(avg_distance_error, axis=0))
