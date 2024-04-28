import collections
import datetime

import functools
import tensorflow as tf

TrainStepInformation = collections.namedtuple(
        'TrainStepInformation', ['loss_information', 'grad_norm', 'grad_norm_by_source',
                                 'max_grad_norm', 'max_grad_value', 'mean_grad_value',
                                 'variance_grad_value'])

MODEL = None
STRATEGY = None
OPTIMIZER = None
BETA_FN = None
TRAIN_STEPS = 500
CONFIG = None

def ShapeList(x):
  ps = x.get_shape().as_list()
  ts = tf.shape(x)
  return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

@tf.function(reduce_retracing=True)
def _TrainStep(train_iterator, cpu_step):
  gradient_accumulation_steps = CONFIG.get('gradient_accumulation_steps', 1)
  def step_fun(training_datas, cpu_step):
    def _reporting_fun(loss_information, grads):
      grad_norm = functools.reduce(
          lambda x,y: tf.math.add(x, tf.norm(y)), grads, 0.0)
      max_grad_norm = functools.reduce(
          lambda x,y: tf.math.maximum(x, tf.norm(y)), grads, 0.0)
      max_grad_value = functools.reduce(
          lambda x,y: tf.math.maximum(x, tf.math.reduce_max(
            tf.math.abs(y))), grads, 0.0)
      sum_grad_value = functools.reduce(
          lambda x,y: tf.math.add(x, tf.math.reduce_sum(
            tf.math.abs(y))), grads, 0.0)
      sum_grad_size = functools.reduce(
          lambda x,y: tf.math.add(x, tf.size(y)), grads, 0)
      sum_grad_size = tf.cast(sum_grad_size, tf.float32)

      mean_grad_value = sum_grad_value/sum_grad_size
      square_diff_grad_value = functools.reduce(
          lambda x,y: tf.math.add(x, tf.math.reduce_sum(
            tf.math.square(tf.math.abs(y)-mean_grad_value))), grads, 0.0)
      trainable_weights = MODEL.trainable_weights()
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
          max_grad_norm=max_grad_norm,
          max_grad_value=max_grad_value,
          mean_grad_value=mean_grad_value,
          variance_grad_value=square_diff_grad_value/sum_grad_size)

    training_datas_iterator = iter(training_datas)
    with tf.GradientTape() as tape:
      loss_information = MODEL.compute_loss(
          training_data=next(training_datas_iterator),
          training=True,
          beta=BETA_FN(cpu_step))
      loss = loss_information.loss

      for i in tf.range(gradient_accumulation_steps-1):
        loss = tf.math.add(
            loss, 
            MODEL.compute_loss(
              training_data=next(training_datas_iterator),
              training=True,
              beta=BETA_FN(cpu_step+i)).loss)
      loss = loss / gradient_accumulation_steps
    trainable_weights = MODEL.trainable_weights()
    grads = tape.gradient(loss, trainable_weights)
    if 'grad_clip_value' in CONFIG:
      clip_value = CONFIG['grad_clip_value']
      grads = [tf.clip_by_value(x, -1*clip_value, clip_value) for x in grads]
    return (loss_information, grads)

    OPTIMIZER.apply_gradients(zip(grads, trainable_weights))
    return _reporting_fun(loss_information, grads)

  FACTORED_STEPS = TRAIN_STEPS // gradient_accumulation_steps
  for i in tf.range(FACTORED_STEPS-1, dtype=tf.int64):
    STRATEGY.run(step_fun, (
        [next(train_iterator) for i in range(gradient_accumulation_steps)], cpu_step + i*gradient_accumulation_steps))
  return STRATEGY.run(
      step_fun,
      ([next(train_iterator) for i in range(gradient_accumulation_steps)],
       cpu_step + (FACTORED_STEPS - 1)*gradient_accumulation_steps))

  

def Train(ds, shuffle_size, batch_size, prefetch_size,
    pdb_vocab, model, optimizer, save_frequency, write_target,
    tensorboard_target, checkpoint_directory, strategy, beta_fn=lambda cpu_step: 1,
    config=dict(), start_cpu_step=0):
  global MODEL
  global STRATEGY
  global OPTIMIZER
  global BETA_FN
  global CONFIG

  MODEL = model
  STRATEGY = strategy
  OPTIMIZER = optimizer
  BETA_FN = beta_fn
  CONFIG = config

  with STRATEGY.scope():
    ckpt = tf.train.Checkpoint(
        ck_step=tf.Variable(start_cpu_step, dtype=tf.int64),
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
        tf.summary.scalar('max_grad_value', train_step_information[0].max_grad_value, step=cpu_step)
        tf.summary.scalar('mean_grad_value', train_step_information[0].mean_grad_value, step=cpu_step)
        tf.summary.scalar('variance_grad_value', train_step_information[0].variance_grad_value, step=cpu_step)
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
