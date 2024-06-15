import flax
import functools
import jax
import optax

def Train(vae, optimizer, vae_params, pdb_vocab, opt_state, ds,
          tensorboard_target,
          shuffle_size, batch_size, prefetch_size, input_size, num_shards, step=0):
  def _LossFn(vae_params, random_key, training_data):
    loss_information = vae.compute_model_loss(random_key, training_data)
    return (loss_information.loss, loss_information)

  loss_and_grad_fn = jax.value_and_grad(
      _LossFn, argnums=[0], has_aux=True)

  @functools.partial(jax.pmap, axis_name='batch', donate_argnums=(1,2))
  def TrainStep(random_key, vae_params, opt_state, training_data):
    (loss, loss_information), grads = loss_and_grad_fn(
        vae_params, random_key, training_data)
    mean_grad = jax.lax.pmean(grads, 'batch')
    updates, opt_state = optimizer.update(mean_grad, opt_state)
    vae_params = optax.apply_updates(vae_params, updates)
    grad_norm = jax.tree_util.tree_reduce(operator.add, jax.tree_util.tree_map(jnp.linalg.norm, mean_grad))
    return (jax.lax.pmean(loss_information, 'batch'),
            grad_norm, vae_params, opt_state)
  
  vae_params = flax.jax_utils.replicate(vae_params)
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
        tf.summary.scalar('grad_norm', grad_norm[0], step=step)
    if step % 1000 == 0:
      model_loading.SaveModel(
              storage_client=storage_client,
              bucket_name=model_save_bucket,
              blob_name=model_save_blob+'/'+str(step),
              encoder_params=flax.jax_utils.unreplicate(encoder_params),
              conditioner_params=flax.jax_utils.unreplicate(conditioner_params),
              decoder_params=flax.jax_utils.unreplicate(decoder_params),
              opt_state=flax.jax_utils.unreplicate(opt_state))
    step += 1
