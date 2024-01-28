import collections

import functools
import tensorflow as tf

TrainStepInformation = collections.namedtuple(
        'TrainStepInformation', ['loss_information', 'grad_norm', 'grad_norm_by_source'])

@tf.function(reduce_retracing=True)
def _TrainStep(model, optimizer, training_data, beta):
  with tf.GradientTape() as tape:
    loss_information = model.compute_loss(
        training_data=training_data,
        training=True,
        beta=beta)
  trainable_weights = model.trainable_weights()
  grads = tape.gradient(loss_information, trainable_weights)
  optimizer.apply_gradients(zip(grads, trainable_weights))
  grad_norm = functools.reduce(
      lambda x,y: tf.math.add(x, tf.norm(y)), grads, 0.0)
  sources = (
      ['conditioner'] * len(model._conditioner.trainable_weights()) +
      ['decoder'] * len(model._decoder.trainable_weights()) +
      ['encoder'] * len(model._encoder.trainable_weights()))
  grad_norm_by_source = {}
  for s,g in zip(sources, grads):
    if s in grad_norm_by_source:
      grad_norm_by_source[s] = tf.math.add(tf.norm(g), grad_norm_by_source[s])
    else:
      grad_norm_by_source[s] = tf.norm(g)

  return TrainStepInformation(
          loss_information=loss_information,
          grad_norm=grad_norm,
          grad_norm_by_source=grad_norm_by_source)

def Train(ds, shuffle_size, batch_size, prefetch_size,
    pdb_vocab, model, optimizer, save_frequency, write_target,
    tensorboard_target, beta_fn=lambda cpu_step: 1):
  summary_writer = tf.summary.create_file_writer(
          tensorboard_target, max_queue=10000, flush_millis=600000)
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
    beta = beta_fn(cpu_step)
    train_step_information = _TrainStep(model, optimizer, training_data, tf.constant(beta))
    if cpu_step % 100 ==0
      with summary_writer.as_default():
        tf.summary.scalar('loss', train_step_information.loss_information.loss, step=step)
        tf.summary.scalar('loss_beta_1', train_step_information.loss_information.loss_beta_1, step=step)
        tf.summary.scalar('logpx_z', train_step_information.loss_information.logpx_z, step=step)
        tf.summary.scalar('logpz', train_step_information.loss_information.logpz, step=step)
        tf.summary.scalar('logqz_x', train_step_information.loss_information.logqz_x, step=step)
        tf.summary.scalar('diff_mae', train_step_information.loss_information.diff_mae, step=step)
        tf.summary.scalar('grad_norm', train_step_information.grad_norm, step=step)
        for s, g in train_step_information.grad_norm_by_source.items():
          tf.summary.scalar('grad_norm_by_source_' + s, g, step=step)
    if cpu_step == 0:
        model.save('{}/version_{}'.format(write_target, cpu_step))
    elif cpu_step % save_frequency==0:
        model.save_weights('{}/version_{}'.format(write_target, cpu_step))
    cpu_step += 1
