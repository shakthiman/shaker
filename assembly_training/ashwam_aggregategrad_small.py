import tensorflow as tf

from protein_folding import pdb_vocab
from protein_folding import train_ds
from protein_folding.variational_model import model_trainer
from protein_folding.variational_model import model
from protein_folding.variational_model.models import sequence_latent_space_with_hierarchical_encoder

from google.cloud import storage

def BetaAnneal(step):
  max_val_for_period = tf.math.minimum(10.0, 1.0 + 0.4*tf.cast(
      tf.math.floordiv(step, 2000), tf.float32))
  inc_per_step = max_val_for_period/1000
  return tf.math.minimum(max_val_for_period, tf.cast(step%2000, tf.float32)*inc_per_step)

def main ():
  cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          #tpu="local"
                                                                       )
  tf.config.experimental_connect_to_cluster(cluster_resolver)
  tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
  strategy = tf.distribute.TPUStrategy(cluster_resolver)
  print("All devices: ", tf.config.list_logical_devices('TPU'))

  client = storage.Client()
  summary_blob = client.bucket("unreplicated-training-data").blob(
      "pdb_training_examples_summary/data_mar_26-00000-of-00001.avro")
  v = pdb_vocab.PDBVocab(summary_blob)
  config = {'scale_coordinates': False, 'should_do_local_transform': True}

  with strategy.scope():
    variational_model = sequence_latent_space_with_hierarchical_encoder.MODEL_FOR_TRAINING(
        v, config)
    variational_model.load_weights(
        'gs://variational_shaker_models/assembly_based_model_prod_ashwam_cut_value2/version_92000')
    optimizer = tf.keras.optimizers.Adam(clipnorm=5e5, amsgrad=True, gradient_accumulation_steps=8)
  ds = train_ds.GetTFExamples(project='shaker-388116',
                              bucket='unreplicated-training-data',
                              blob_prefix='pdb_training_examples_mar_26/polypeptides',
                              num_parallel_calls=None,
                              cluster_shuffle_size=1000,
                              cluster_cycle_length=1000)
  print('Num Replicas: ', strategy.num_replicas_in_sync)
  
  model_trainer.Train(
    ds=ds,
    shuffle_size=10000,
    batch_size=2*strategy.num_replicas_in_sync,
    prefetch_size=150,
    pdb_vocab=pdb_vocab.PDBVocab(summary_blob),
    model=variational_model,
    optimizer=optimizer,
    save_frequency=1000,
    write_target='gs://variational_shaker_models/assembly_based_model_prod_ashwam_aggregategrad_small2',
    tensorboard_target='gs://variational_shaker_models/tensorboard/assembly_based_model_prod_ashwam_aggregategrad_small2',
    checkpoint_directory='gs://variational_shaker_models/checkpoints/assembly_based_model_prod_ashwam_aggregategrad_small',
    strategy=strategy,
    beta_fn= lambda cpu_step: BetaAnneal(cpu_step),
    config={'grad_clip_value': 10},
    start_cpu_step=92000)

if __name__ == "__main__":
  main()
