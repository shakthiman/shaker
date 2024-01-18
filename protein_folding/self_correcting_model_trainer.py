import collections
import tensorflow as tf
from protein_folding import multi_diffusion_model

import functools

TrainingSample = collections.namedtuple(
    'TrainingSample', ['z_t', 'g_t', 't', 'g_t_1', 'eps_t'])

def _TrainingSamples(base_model, f, f_mask, cond, t, T, steps):
  g_t = base_model.gamma(t)
  g_t_1 = base_model.gamma(t - (1.0/T))
  eps = tf.random.normal(tf.shape(f))
  z_t = base_model.variance_preserving_map(f, g_t, eps)

  z_ts = [z_t]
  g_ts = [g_t]
  ts = [t]
  g_t_1s = [g_t_1]
  eps_ts = [eps]

  s = t
  z_s = z_t
  for step in steps:
    t = s
    z_t = z_s

    s = s - step
    # Simulate forward one step.
    z_s = base_model.sample_step_vec(t, s, z_t, cond, f_mask)
    g_s = base_model.gamma(s)
    g_s_1 = base_model.gamma(s -(1.0/T))
    eps_s = base_model.perfect_score_vec(z_s, f, g_s)

    z_ts.append(z_s)
    g_ts.append(g_s)
    ts.append(s)
    g_t_1s.append(g_s_1)
    eps_ts.append(eps_s)

  return TrainingSample(
      z_t=tf.concat(z_ts, 0),
      g_t=tf.concat(g_ts, 0),
      t=tf.concat(ts, 0),
      g_t_1=tf.concat(g_t_1s, 0),
      eps_t=tf.concat(eps_ts, 0))

@tf.function(reduce_retracing=True)
def _train_step(base_model, train_model, training_data, optimizer):
  x = training_data['normalized_coordinates']
  x_mask = multi_diffusion_model.XMask(x)
  cond = base_model.conditioning(training_data, training=False)
  f = base_model.encode(training_data, cond, training=False)

  n_batch = tf.shape(x)[0]
  t0 = tf.random.uniform(shape=[])
  def _GetTs(min_val, max_val):
    return tf.math.floormod(t0+tf.range(0, 1, 1./tf.cast(n_batch, 'float32'),
      dtype='float32'), max_val-min_val) + min_val

  ts_and_steps = [
      (_GetTs(0.9, 1.0), [0.1]*9),
      (_GetTs(0.5, 1.0), [0.1]*5),
      (_GetTs(0.0, 1.0), [])]

  training_samples = [
      _TrainingSamples(base_model, f, x_mask, cond, t, train_model.timesteps(), s)
      for t,s in ts_and_steps]

  training_sample = TrainingSample(
      z_t=tf.concat([ts.z_t for ts in training_samples], 0),
      g_t=tf.concat([ts.g_t for ts in training_samples], 0),
      t=tf.concat([ts.t for ts in training_samples], 0),
      g_t_1=tf.concat([ts.g_t_1 for ts in training_samples], 0),
      eps_t=tf.concat([ts.eps_t for ts in training_samples], 0))

  batch_indxs = tf.random.shuffle(tf.range(tf.shape(training_sample.z_t)[0]))
  x_masks = tf.concat([x_mask]*(10 + 6 + 1), 0)
  conds = tf.concat([cond]*(10 + 6 + 1), 0)

  split_indx = tf.split(batch_indxs, 17)

  for split in split_indx:
    with tf.GradientTape() as tape:
      smaller_sample = TrainingSample(
          z_t=tf.gather(training_sample.z_t, split),
          g_t=tf.gather(training_sample.g_t, split),
          t=tf.gather(training_sample.t, split),
          g_t_1=tf.gather(training_sample.g_t_1, split),
          eps_t=tf.gather(training_sample.eps_t, split))
      small_z_mask = tf.gather(x_masks, split)
      small_conds = tf.gather(conds, split)
      eps_hat = train_model.score(
              z_t=smaller_sample.z_t,
              z_mask=small_z_mask,
              gamma=smaller_sample.g_t,
              cond=small_conds,
              training=True)
      loss_diff, loss_diff_mse = train_model.diffusion_loss_from_eps(
              eps=smaller_sample.eps_t,
              eps_hat=eps_hat,
              t=smaller_sample.t,
              T=train_model.timesteps(),
              f_mask=small_z_mask,
              normalize_by_num_atoms=False)
      trainable_weights = train_model.scorer_weights()
      grads = tape.gradient(loss_diff, trainable_weights)
      optimizer.apply_gradients(zip(grads, trainable_weights))
  grad_norm = functools.reduce(lambda x,y: tf.math.add(x, tf.norm(y)),
      grads, 0.0)
  return (tf.reduce_mean(loss_diff), tf.reduce_mean(loss_diff_mse), grad_norm)

def TrainMultiChainModel(ds, shuffle_size, batch_size, prefetch_size,
    save_frequency, pdb_vocab, base_model, train_model, optimizer,
    write_target, tensorboard_target):
  summary_writer = tf.summary.create_file_writer(tensorboard_target)
  tds = ds.shuffle(shuffle_size).map(
      lambda x: {
        'residue_names': pdb_vocab.GetResidueNamesId(x['resname'].to_tensor()),
        'atom_names': pdb_vocab.GetResidueNamesId(x['atom_name'].to_tensor()),
        'normalized_coordinates': x['atom_coords'].to_tensor()}).padded_batch(
            batch_size,
            padded_shapes={
              'residue_names': [None, None],
              'atom_names': [None, None],
              'normalized_coordinates': [None, None, 3]}).prefetch(prefetch_size)
  cpu_step = 0
  for step, training_data in tds.enumerate():
    loss, loss_diff_mse, grad_norm = _train_step(base_model, train_model, training_data, optimizer)
    with summary_writer.as_default():
      tf.summary.scalar('loss', loss, step=step)
      tf.summary.scalar('loss_diff_mse', loss_diff_mse, step=step)
      tf.summary.scalar('grad_norm', grad_norm, step=step)
    if cpu_step == 0:
      train_model.save('{}/version_{}'.format(write_target, step))
    elif cpu_step % save_frequency == 0:
      train_model.save_weights('{}/version_{}'.format(write_target, step))
      # Hack to update the base_model
      base_model._scorer._model.set_weights(
          train_model._scorer._model.get_weights())
    cpu_step += 1
