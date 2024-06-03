import flax
import functools
import jax
from jax import random
from jax import numpy as jnp
import optax
import operator


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

def Train(storage_client, ds, shuffle_size, batch_size, input_size, prefetch_size, num_shards, pdb_vocab, random_key,
          model_save_bucket, model_save_blob, tensorboard_target, encoder_model, conditioner, decoder_model, optimizer, compute_loss_fn,
          encoder_params, conditioner_params, decoder_params, opt_state):
  def _LossFn(rk, ep, cp, dp, td):
    loss_information = compute_loss_fn(
        random_key=rk,
        encoder_model=encoder_model,
        conditioner_model=conditioner,
        decoder_model=decoder_model,
        encoder_params=ep,
        conditioner_params=cp,
        decoder_params=dp,
        training_data=td)
    return (loss_information.loss, loss_information)

  loss_and_grad_fn = jax.value_and_grad(
      _LossFn, argnums=[1,2,3], has_aux=True)

  @functools.partial(jax.pmap, axis_name='batch', donate_argnums=(1,2,3,4))
  def TrainStep(rk, ep, cp, dp, os, td):
    (loss,loss_information), grads = loss_and_grad_fn(
        rk, ep, cp, dp, td)
    mean_grad = jax.lax.pmean(grads, 'batch')
    updates, os = optimizer.update(mean_grad, os)
    (ep, cp, dp) = optax.apply_updates(
        (ep, cp, dp), updates)
    grad_norm = jax.tree_util.tree_reduce(operator.add, jax.tree_util.tree_map(jnp.linalg.norm, mean_grad))
    return (jax.lax.pmean(loss_information, 'batch'),
            grad_norm,
            (ep, cp, dp), os)

  encoder_params = flax.jax_utils.replicate(encoder_params)
  conditioner_params = flax.jax_utils.replicate(conditioner_params)
  decoder_params = flax.jax_utils.replicate(decoder_params)
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
  step = 0
  for t in tds:
    training_data = {k:v.numpy() for k,v in t.items()}
    random_key, loss_key = random.split(random_key, 2)
    loss_keys = random.split(loss_key, num=(num_shards,))
    print('start')
    (loss_information,
     grad_norm,
     (encoder_params, conditioner_params, decoder_params),
     opt_state) = TrainStep(
         loss_keys, encoder_params, conditioner_params,
         decoder_params, opt_state, training_data)
    print('finish')
    if step % 100 == 0:
      with summary_writer.as_default():
        tf.summary.scalar('loss', loss_information.loss[0], step=step)
        tf.summary.scalar('loss_beta_1', loss_information.loss_beta_1[0], step=step)
        tf.summary.scalar('logpx_z', loss_information.logpx_z[0], step=step)
        tf.summary.scalar('logpz', loss_information.logpz[0], step=step)
        tf.summary.scalar('logqz_x', loss_information.logqz_x[0], step=step)
        tf.summary.scalar('diff_mae', loss_information.diff_mae[0], step=step)
        tf.summary.scalar('grad_norm', grad_norm[0], step=step)
      model_loading.SaveModel(
              storage_client=storage_client,
              bucket_name=model_save_bucket,
              blob_name=model_save_blob+'/'+str(step),
              encoder_params=flax.jax_utils.unreplicate(encoder_params),
              conditioner_params=flax.jax_utils.unreplicate(conditioner_params),
              decoder_params=flax.jax_utils.unreplicate(decoder_params))
    step += 1
