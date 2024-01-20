import tensorflow as tf

def _TrainStep(model, optimizer, training_data):
  with tf.GradientTape() as tape:
    loss = model.compute_loss(training_data, True)
  trainable_weights = model.trainable_weights()
  grads = tape.gradient(loss, trainable_weights)
  optimizer.apply_gradients(zip(grads, trainable_weights))
  return loss

def Train(ds, shuffle_size, batch_size, prefetch_size,
    pdb_vocab, model, optimizer):
  tds = ds.shuffle(shuffle_size).map(
      lambda x: {
        'residue_names':pdb_vocab.GetResidueNamesId(x['resname']).to_tensor(),
        'atom_names':pdb_vocab.GetResidueNamesId(x['atom_name']).to_tensor(),
        'normalized_coordinates': x['atom_coords'].to_tensor()}).padded_batch(
            batch_size,
            padded_shapes={
              'residue_names': [None, None],
              'atom_names': [None, None],
              'normalized_coordinates': [None, None, 3]}).prefetch(prefetch_size)
  cpu_step = 0
  for step, training_data in tds.enumerate():
    loss = _TrainStep(model, optimizer, training_data)
    if cpu_step%10==0:
      print('Step: ', cpu_step)
      print('Loss: ', loss)
