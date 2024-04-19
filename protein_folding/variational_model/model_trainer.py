import collections
import datetime

import functools
import tensorflow as tf

TrainStepInformation = collections.namedtuple(
        'TrainStepInformation', ['loss_information', 'grad_norm', 'grad_norm_by_source', 'max_grad_norm'])

MODEL = None
STRATEGY = None
OPTIMIZER = None
BETA_FN = None
TRAIN_STEPS = 100

@tf.function(reduce_retracing=True)
def _TrainStep(train_iterator, cpu_step):
  def step_fun(training_data, beta):
    with tf.GradientTape() as tape:
      loss_information = MODEL.compute_loss(
          training_data=training_data,
          training=True,
          beta=beta)
    trainable_weights = MODEL.trainable_weights()
    grads = tape.gradient(loss_information, trainable_weights)
    OPTIMIZER.apply_gradients(zip(grads, trainable_weights))
    grad_norm = functools.reduce(
        lambda x,y: tf.math.add(x, tf.norm(y)), grads, 0.0)
    max_grad_norm = functools.reduce(
        lambda x,y: tf.math.maximum(x, tf.norm(y)), grads, 0.0)
    sources = (
        ['conditioner'] * len(MODEL._conditioner.trainable_weights()) +
        ['decoder'] * len(MODEL._decoder.trainable_weights()) +
        ['encoder'] * len(MODEL._encoder.trainable_weights()) +
        ['rotation_model'] * len(MODEL._rotation_model.trainable_weights) +
        ['local_transformation_model'] * len(
            MODEL._local_transformation_model.trainable_weights()))
    grad_norm_by_source = {}
    for s,g in zip(sources, grads):
      if s in grad_norm_by_source:
        grad_norm_by_source[s] = tf.math.add(tf.norm(g), grad_norm_by_source[s])
      else:
        grad_norm_by_source[s] = tf.norm(g)

    return TrainStepInformation(
            loss_information=loss_information,
            grad_norm=grad_norm,
            grad_norm_by_source=grad_norm_by_source,
            max_grad_norm=max_grad_norm)
  for i in tf.range(TRAIN_STEPS-1, dtype=tf.int64):
    STRATEGY.run(step_fun, (next(train_iterator), BETA_FN(cpu_step + i)))
  return STRATEGY.run(step_fun, (next(train_iterator), BETA_FN(cpu_step + TRAIN_STEPS - 1)))

def Train(ds, shuffle_size, batch_size, prefetch_size,
    pdb_vocab, model, optimizer, save_frequency, write_target,
    tensorboard_target, checkpoint_directory, strategy, beta_fn=lambda cpu_step: 1):
  global MODEL
  global STRATEGY
  global OPTIMIZER
  global BETA_FN

  MODEL = model
  STRATEGY = strategy
  OPTIMIZER = optimizer
  BETA_FN = beta_fn

  with STRATEGY.scope():
    ckpt = tf.train.Checkpoint(
        ck_step=tf.Variable(0, dtype=tf.int64),
        optimizer=OPTIMIZER,
        conditioner=MODEL._conditioner._model,
        decoder=MODEL._decoder._model,
        encoder=MODEL._encoder._model,
        rotation_model=MODEL._rotation_model)
  manager = tf.train.CheckpointManager(ckpt, checkpoint_directory, max_to_keep=3)
  ckpt.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")

  summary_writer = tf.summary.create_file_writer(
          tensorboard_target, max_queue=10000, flush_millis=600000)
  tds = ds.shuffle(shuffle_size).map(
      lambda x: {
        'residue_names':pdb_vocab.GetResidueNamesId(x['resname']).to_tensor(),
        'atom_names':pdb_vocab.GetAtomNamesId(x['atom_name']).to_tensor(),
        'normalized_coordinates': x['atom_coords'].to_tensor()}).padded_batch(
            batch_size,
            padded_shapes={
              'residue_names': [4, 6000],
              'atom_names': [4, 6000],
              'normalized_coordinates': [4, 6000, 3]}).prefetch(prefetch_size)
  tds = STRATEGY.experimental_distribute_dataset(tds)
  train_iterator = iter(tds)
  cpu_step = ckpt.ck_step.numpy()
  while True:
    print('Wall Time Start: ', datetime.datetime.now())
    train_step_information = _TrainStep(train_iterator, tf.constant(cpu_step))
    train_step_information = STRATEGY.experimental_local_results(train_step_information)
    print(cpu_step)
    print('Wall Time Finish: ', datetime.datetime.now())
    if cpu_step % 100 == 0:
      with summary_writer.as_default():
        tf.summary.scalar('loss', train_step_information[0].loss_information.loss, step=cpu_step)
        tf.summary.scalar('loss_beta_1', train_step_information[0].loss_information.loss_beta_1, step=cpu_step)
        tf.summary.scalar('local_logpx_z', train_step_information[0].loss_information.local_logpx_z, step=cpu_step)
        tf.summary.scalar('logpx_z', train_step_information[0].loss_information.logpx_z, step=cpu_step)
        tf.summary.scalar('logpz', train_step_information[0].loss_information.logpz, step=cpu_step)
        tf.summary.scalar('logqz_x', train_step_information[0].loss_information.logqz_x, step=cpu_step)
        tf.summary.scalar('diff_mae', train_step_information[0].loss_information.diff_mae, step=cpu_step)
        tf.summary.scalar('local_diff_mae', train_step_information[0].loss_information.local_diff_mae, step=cpu_step)
        tf.summary.scalar('grad_norm', train_step_information[0].grad_norm, step=cpu_step)
        tf.summary.scalar('max_grad_norm', train_step_information[0].max_grad_norm, step=cpu_step)
        for s, g in train_step_information[0].grad_norm_by_source.items():
          tf.summary.scalar('grad_norm_by_source_' + s, g, step=cpu_step)
    if cpu_step == 0:
        MODEL.save('{}/version_{}'.format(write_target, cpu_step))
    elif cpu_step % save_frequency == 0:
        MODEL.save_weights('{}/version_{}'.format(write_target, cpu_step))
        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(int(cpu_step), save_path))
    cpu_step += TRAIN_STEPS
    ckpt.ck_step.assign(cpu_step)
