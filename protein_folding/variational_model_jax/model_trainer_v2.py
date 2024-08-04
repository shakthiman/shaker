import flax
import functools
import jax
from jax import numpy as jnp
from jax import random
import operator
import optax
import tensorflow as tf

from protein_folding.variational_model_jax import model_loading

def Featurize(x, pdb_vocab):
  residue_names = pdb_vocab.GetResidueNamesId(x['resname'])
  atom_names = pdb_vocab.GetAtomNamesId(x['atom_name'])
  normalized_coordinates = x['atom_coords']

  residue_names = residue_names.to_tensor()
  atom_names = atom_names.to_tensor()
  normalized_coordinates = normalized_coordinates.to_tensor()

  peptide_indx = tf.expand_dims(tf.range(tf.shape(residue_names)[0], dtype=tf.int64), -1) * tf.ones_like(residue_names)
  atom_indx = tf.expand_dims(tf.range(tf.shape(residue_names)[1], dtype=tf.int64), 0) * tf.ones_like(residue_names)

  return {
    #'structure_id': x['structure_id'],
    'peptide_indices': tf.reshape(peptide_indx, [-1]),
    'atom_indices': tf.reshape(atom_indx, [-1]),
    'residue_names': tf.reshape(residue_names, [-1]),
    'atom_names': tf.reshape(atom_names, [-1]),
    'normalized_coordinates': tf.reshape(normalized_coordinates, [-1, 3])
  }

def Train(storage_client, ds, shuffle_size, batch_size, input_size,
          prefetch_size, num_shards, pdb_vocab, random_key, model_save_bucket,
          model_save_blob, tensorboard_target, vae, optimizer, vae_params,
          opt_state, step=0):
  def _LossFn(vae_params, random_key, training_data):
    loss_key, dropout_key = random.split(random_key, 2)
    loss_information = vae.apply(
            vae_params, loss_key, training_data,
            method=vae.compute_model_loss,
            rngs={'dropout': dropout_key})
    return (loss_information.loss, loss_information)

  loss_and_grad_fn = jax.value_and_grad(
      _LossFn, argnums=0, has_aux=True)

  @functools.partial(jax.pmap, axis_name='batch', donate_argnums=(1,2))
  def TrainStep(random_key, vae_params, opt_state, training_data):
    (loss, loss_information), grads = loss_and_grad_fn(
        vae_params, random_key, training_data)
    mean_grad = jax.lax.pmean(grads, 'batch')
    updates, opt_state = optimizer.update(mean_grad, opt_state, vae_params)
    vae_params = optax.apply_updates(vae_params, updates)
    grad_norm = jax.tree_util.tree_reduce(operator.add, jax.tree_util.tree_map(jnp.linalg.norm, mean_grad))
    return (jax.lax.pmean(loss_information, 'batch'),
            grad_norm, vae_params, opt_state)
  
  vae_params = flax.jax_utils.replicate(vae_params)
  opt_state = flax.jax_utils.replicate(opt_state)
  tds = ds.shuffle(shuffle_size).map(
      lambda x: Featurize(x, pdb_vocab)).padded_batch(
          batch_size,
          padded_shapes={
            'peptide_indices': [input_size],
            'atom_indices': [input_size],
            'residue_names': [input_size],
            'atom_names': [input_size],
            'normalized_coordinates': [input_size, 3]
            }).batch(num_shards).prefetch(prefetch_size)
  summary_writer = tf.summary.create_file_writer(
      tensorboard_target, max_queue=10000, flush_millis=600000)
  for t in tds:
    training_data = {k:v.numpy() for k,v in t.items()}
    random_key, loss_key = random.split(random_key, 2)
    loss_keys = random.split(loss_key, num=(num_shards,))
    print('start')
    (loss_information, grad_norm, vae_params, opt_state) = TrainStep(
        loss_keys, vae_params, opt_state, training_data)
    print('finish')
    if step % 100 == 0:
      with summary_writer.as_default():
        tf.summary.scalar('loss', loss_information.loss[0], step=step)
        tf.summary.scalar('loss_beta_1', loss_information.loss_beta_1[0], step=step)
        tf.summary.scalar('logpx_z', loss_information.logpx_z[0], step=step)
        tf.summary.scalar('logpz', loss_information.logpz[0], step=step)
        tf.summary.scalar('logqz_x', loss_information.logqz_x[0], step=step)
        tf.summary.scalar('diff_mae', loss_information.diff_mae[0], step=step)
        tf.summary.scalar('loss_alpha_carbon_clash',
                          loss_information.loss_alpha_carbon_clash[0],
                          step=step)
        tf.summary.scalar('num_hard_clashes',
                          loss_information.num_hard_clashes[0], step=step)
        tf.summary.scalar('num_soft_clashes',
                          loss_information.num_soft_clashes[0], step=step)
        tf.summary.scalar('clash_sum_squares',
                          loss_information.clash_sum_squares[0], step=step)
        tf.summary.scalar('loss_dihedral_loss',
                          loss_information.loss_dihedral_loss[0], step=step)
        tf.summary.scalar('total_phi_error',
                          loss_information.dihedral_loss.total_phi_error[0], step=step)
        tf.summary.scalar('total_psi_error',
                          loss_information.dihedral_loss.total_psi_error[0], step=step)
        tf.summary.scalar('total_omega_error',
                          loss_information.dihedral_loss.total_omega_error[0], step=step)
        tf.summary.scalar('average_phi_error',
                          loss_information.dihedral_loss.average_phi_error[0], step=step)
        tf.summary.scalar('average_psi_error',
                          loss_information.dihedral_loss.average_psi_error[0], step=step)
        tf.summary.scalar('average_omega_error',
                          loss_information.dihedral_loss.average_omega_error[0], step=step)
        tf.summary.scalar(
            'alpha_carbon_only_radius_of_gyration_diff',
            loss_information
            .radius_of_gyration_loss
            .alpha_carbon_only_radius_of_gyration_diff[0],
            step=step)
        tf.summary.scalar(
            'alpha_carbon_only_radius_of_gyration_squared_diff',
            loss_information
            .radius_of_gyration_loss
            .alpha_carbon_only_radius_of_gyration_squared_diff[0],
            step=step)
        tf.summary.scalar(
            'alpha_carbon_squared_distances_loss',
            loss_information
            .distance_matrix_loss
            .alpha_carbon_squared_distances_loss[0],
            step=step)
        tf.summary.scalar(
            'alpha_carbon_avg_distance_error',
            loss_information
            .distance_matrix_loss
            .alpha_carbon_avg_distance_error[0],
            step=step)
        tf.summary.scalar('grad_norm', grad_norm[0], step=step)
    if step % 5000 == 0:
      model_loading.SaveModelV2(
              storage_client=storage_client,
              bucket_name=model_save_bucket,
              blob_name=model_save_blob+'/'+str(step),
              vae_params=flax.jax_utils.unreplicate(vae_params),
              opt_state=flax.jax_utils.unreplicate(opt_state))
    step += 1
