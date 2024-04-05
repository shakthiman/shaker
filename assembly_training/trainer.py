import tensorflow as tf

from protein_folding import pdb_vocab
from protein_folding import train_ds
from protein_folding.variational_model import model_trainer
from protein_folding.variational_model.models import sequence_latent_space_with_hierarchical_encoder

from google.cloud import storage

def BetaAnneal(step):
  return min(10.0, tf.cast(step%2000, tf.float32)*1e-2)

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

  with strategy.scope():
    variational_model = sequence_latent_space_with_hierarchical_encoder.MODEL_FOR_TRAINING(v)
    optimizer = tf.keras.optimizers.Adam(clipnorm=100)
    ds = train_ds.GetTFExamples(project='shaker-388116',
                                bucket='unreplicated-training-data',
                                blob_prefix='pdb_training_examples_mar_26/polypeptides',
                                num_parallel_calls=None,
                                cluster_shuffle_size=1000,
                                cluster_cycle_length=1000)
    model_trainer.Train(
      ds=ds,
      shuffle_size=10,
      batch_size=4,
      prefetch_size=10,
      pdb_vocab=pdb_vocab.PDBVocab(summary_blob),
      model=variational_model,
      optimizer=optimizer,
      save_frequency=500,
      write_target='gs://variational_shaker_models/assembly_based_model_test2',
      tensorboard_target='gs://variational_shaker_models/tensorboard/assembly_based_model_test2',
      checkpoint_directory='gs://variational_shaker_models/checkpoints/assembly_based_model_test2',
      strategy=strategy,
      beta_fn= lambda cpu_step: BetaAnneal(cpu_step))

if __name__ == "__main__":
  main()
