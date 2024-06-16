from jax import random
import tensorflow as tf

from protein_folding import pdb_vocab
from protein_folding import train_ds
from protein_folding.variational_model_jax import model_trainer_v2
from protein_folding.variational_model_jax.models import dropped_out_decoder
from protein_folding.variational_model_jax import model_loading

from google.cloud import storage
import optax

_INPUT_SIZE = 24000
_BATCH_SIZE = 2
_NUM_BLOCKS = 150

def main ():
  client = storage.Client()
  summary_blob = client.bucket("unreplicated-training-data").blob(
      "pdb_training_examples_summary/data_mar_26-00000-of-00001.avro")
  v = pdb_vocab.PDBVocab(summary_blob)
  ds = train_ds.GetTFExamples(project='shaker-388116',
                              bucket='unreplicated-training-data',
                              blob_prefix='pdb_training_examples_mar_26/polypeptides',
                              num_parallel_calls=None,
                              cluster_shuffle_size=1000,
                              cluster_cycle_length=1000)

  random_key = random.key(1)
  vae = dropped_out_decoder.GetModel(
          batch_size=_BATCH_SIZE,
          input_length=_INPUT_SIZE,
          num_blocks=_NUM_BLOCKS,
          pdb_vocab=v,
          deterministic=False)
  optimizer = optax.chain(
          optax.clip_by_global_norm(5e5),
          optax.adam(1e-3))
  optimizer = optax.MultiSteps(optimizer, every_k_schedule=32)
  random_key, model_init_key = random.split(random_key, 2)
  vae_params = dropped_out_decoder.Init(
          random_key=model_init_key,
          vae=vae,
          batch_size=_BATCH_SIZE,
          input_length=_INPUT_SIZE)
  opt_state = optimizer.init(vae_params)

  random_key, train_key = random.split(random_key, 2)
  model_trainer_v2.Train(
    storage_client=client,
    ds=ds,
    shuffle_size=10000,
    batch_size=_BATCH_SIZE,
    input_size=_INPUT_SIZE,
    prefetch_size=150,
    num_shards=8,
    pdb_vocab=v,
    random_key=random_key,
    model_save_bucket='variational_shaker_models',
    model_save_blob='assembly_based_jax_mahat',
    tensorboard_target='gs://variational_shaker_models/tensorboard/assembly_based_jax_mahat',
    vae=vae,
    optimizer=optimizer,
    vae_params=vae_params,
    opt_state=opt_state,
    step=0)

if __name__ == "__main__":
  main()
