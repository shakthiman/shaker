import tensorflow as tf

import functools

@tf.function(reduce_retracing=True)
def _SingleChainTrainStep(training_data, model, optimizer):
  with tf.GradientTape() as tape:
    l1, l2, l3, loss_diff_mse, recon_diff = model.compute_model_loss(
        training_data)
    loss = tf.reduce_mean(l1+l2+l3)
  trainable_weights = model.trainable_weights()
  grads = tape.gradient(loss, trainable_weights)
  optimizer.apply_gradients(zip(grads, trainable_weights))
  return (
      tf.reduce_mean(l1), tf.reduce_mean(l2), tf.reduce_mean(l3),
      tf.reduce_mean(loss_diff_mse), recon_diff)

@tf.function(reduce_retracing=True)
def _MultiChainTrainStep(training_data, model, optimizer):
  with tf.GradientTape() as tape:
    l1, l2, l3, loss_diff_mse, recon_diff = model.compute_model_loss(training_data)
    loss = tf.reduce_mean(l1+l2+l3)
  trainable_weights = model.trainable_weights()
  grads = tape.gradient(loss, trainable_weights)
  grad_norm = functools.reduce(lambda x,y: tf.math.add(x, tf.norm(y)),
      grads, 0.0)
  tf.cond(tf.math.is_nan(grad_norm), lambda: 0, lambda: optimizer.apply_gradients(zip(grads, trainable_weights)))
  return (
      tf.reduce_mean(l1), tf.reduce_mean(l2), tf.reduce_mean(l3),
      tf.reduce_mean(loss_diff_mse), recon_diff, grad_norm)

def TrainSingleChainModel(ds,
    shuffle_size, batch_size, prefetch_size, pdb_vocab, model, optimizer,
    write_target):
  tds = ds.shuffle(shuffle_size).map(
      lambda x:{
        'residue_names': pdb_vocab.GetResidueNamesId(x['resname'].to_tensor()[0]),
        'atom_names': pdb_vocab.GetAtomNamesId(x['atom_name'].to_tensor()[0]),
        'normalized_coordinates': x['atom_coords'].to_tensor()[0]
        }).padded_batch(
            batch_size,
            padded_shapes={
              'residue_names': [None],
              'atom_names': [None],
              'normalized_coordinates': [None, 3],
              }).prefetch(prefetch_size)
  for step, training_data in tds.enumerate():
    l1, l2, l3, loss_diff_mse, recon_diff = _SingleChainTrainStep(
            training_data, model, optimizer)
    if step % 10==0:
      #print("Training Loss (for one batch) at step %d: %.4f"
      #  % (step, float(tf.reduce_mean(l1+l2+l3))))
      print("Training L1 Loss (for one batch) at step %d: %.4f"
                  % (step, l1))
      print("Training L2 Loss (for one batch) at step %d: %.4f"
                  % (step, l2))
      print("Training L3 Loss (for one batch) at step %d: %.4f"
                  % (step, l3))
      print("loss_diff_mse (for one batch) at step %d: %.4f"
                  % (step, loss_diff_mse))
      print("recon_diff (for one batch) at step %d: %.4f"
                  % (step, float(recon_diff)))
      #print("l1: ", l1)
      #print("l2: ", l2)
      #print("l3: ", l3)
      print("Seen so far: %s samples" % ((step + 1) * batch_size))
    if step % 200==0:
      model.save('{}/version_{}'.format(write_target, step))

def TrainMultiChainModel(ds, shuffle_size, batch_size, prefetch_size,
    save_frequency, pdb_vocab, model, optimizer, write_target, tensorboard_target):
  summary_writer = tf.summary.create_file_writer(tensorboard_target)
  tds = ds.shuffle(shuffle_size).map(
      lambda x: {
        'residue_names': pdb_vocab.GetResidueNamesId(x['resname'].to_tensor()),
        'atom_names': pdb_vocab.GetAtomNamesId(x['atom_name'].to_tensor()),
        'normalized_coordinates': x['atom_coords'].to_tensor()}).padded_batch(
            batch_size,
            padded_shapes={
              'residue_names': [None, None],
              'atom_names': [None, None],
              'normalized_coordinates': [None, None, 3]}).prefetch(prefetch_size)
  cpu_step = 0
  for step, training_data in tds.enumerate():
    l1, l2, l3, loss_diff_mse, recon_diff, grad_norm = _MultiChainTrainStep(
            training_data, model, optimizer)
    with summary_writer.as_default():
      tf.summary.scalar('l1_loss', l1, step=step)
      tf.summary.scalar('l2_loss', l2, step=step)
      tf.summary.scalar('l3_loss', l3, step=step)
      tf.summary.scalar('loss_diff_mse', loss_diff_mse, step=step)
      tf.summary.scalar('recon_diff', recon_diff, step=step)
      tf.summary.scalar('training_data_dim_0', tf.shape(training_data['residue_names'])[0], step=step)
      tf.summary.scalar('training_data_dim_1', tf.shape(training_data['residue_names'])[1], step=step)
      tf.summary.scalar('training_data_dim_2', tf.shape(training_data['residue_names'])[2], step=step)
      tf.summary.scalar('grad_norm', grad_norm, step=step)
    if cpu_step%10==0:
      print(cpu_step)
    if cpu_step == 0:
      model.save('{}/version_{}'.format(write_target, step))
    elif cpu_step % save_frequency==0:
      model.save_weights('{}/version_{}'.format(write_target, step))
    cpu_step += 1
