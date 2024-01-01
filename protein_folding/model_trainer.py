import tensorflow as tf

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
  optimizer.apply_gradients(zip(grads, trainable_weights))
  return (
      tf.reduce_mean(l1), tf.reduce_mean(l2), tf.reduce_mean(l3),
      tf.reduce_mean(loss_diff_mse), recon_diff)

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

def TrainMultiChainModel(ds, shuffle_size, batch_size, prefetch_size, pdb_vocab, model, optimizer, write_target):
  def _IgnoreCondition(x):
    peptide_shapes = tf.map_fn(lambda y: tf.shape(y)[0], x['resname'],
        fn_output_signature=tf.int32)
    return tf.math.reduce_min(peptide_shapes)==0
  tds = ds.shuffle(shuffle_size).filter(lambda x: not _IgnoreCondition(x)).map(
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
    l1, l2, l3, loss_diff_mse, recon_diff = _MultiChainTrainStep(training_data, model, optimizer)
    if cpu_step%10==0:
      print(cpu_step)
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
    #if step % 200==0:
    #  model.save('{}/version_{}'.format(write_target, step))
    cpu_step += 1
