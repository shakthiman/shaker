import tensorflow as tf

@tf.function(reduce_retracing=True)
def _SingleChainTrainStep(training_data, model):
  with tf.GradientTape() as tape:
    l1, l2, l3, loss_diff_mse, recon_diff = model.compute_model_loss(
        training_data)
    loss = tf.reduce_mean(l1+l2+l3)
  trainable_weights = model.trainable_weights()
  grads = tape.gradient(loss, trainable_weights)
  optimizer.apply_gradient(zip(grads, trainable_weights))
  return (
      tf.reduce_mean(l1), tf.reduce_mean(l2), tf.reduce_mean(l3),
      tf.reduce_mean(loss_diff_mse), recon_diff)

def TrainSingleChainModel(train_ds,
    shuffle_size, batch_size, prefetch_size, model):
  train_ds.shuffle(shuffle_size).map(
      lambda x:{
        'residue_names': x['resname'][0],
        'atom_names': x['atom_name'][0],
        'normalized_coordinates': x['atom_coords'][0]
        }).padded_batch(
            batch_size,
            padded_shapes={
              'residue_names': [None],
              'atom_names': [None],
              'normalized_coordinates': [None, 3],
              }).prefetch(10)
  for step, training_data in train_ds.enumerate():
    l1, l2, l3, loss_diff_mse, recon_diff = train_step(training_data)
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
