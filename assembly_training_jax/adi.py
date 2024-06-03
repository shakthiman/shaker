from jax import random
import tensorflow as tf

from protein_folding import pdb_vocab
from protein_folding import train_ds
from protein_folding.variational_model_jax import model_trainer
from protein_folding.variational_model_jax.models import first_model

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

  random_key = random.key(0)
  (encoder_model, conditioner, decoder_model) = first_model.GetModels(
          batch_size=_BATCH_SIZE,
          input_length=_INPUT_SIZE,
          num_blocks=_NUM_BLOCKS,
          pdb_vocab=v)
  optimizer = optax.adam(1e-3)
  optimizer = optax.MultiSteps(optimizer, every_k_schedule=32)
  random_key, model_init_key = random.split(random_key, 2)
  (encoder_params, conditioner_params, decoder_params) = first_model.Init(
          random_key=model_init_key,
          encoder=encoder_model,
          decoder=decoder_model,
          conditioner=conditioner,
          batch_size=_BATCH_SIZE,
          input_length=_INPUT_SIZE)
  opt_state = optimizer.init((encoder_params, conditioner_params, decoder_params))

  random_key, train_key = random.split(random_key, 2)
  model_trainer.Train(
    ds=ds,
    shuffle_size=10000,
    batch_size=_BATCH_SIZE,
    input_size=_INPUT_SIZE,
    prefetch_size=150,
    num_shards=8,
    pdb_vocab=v,
    random_key=random_key,
    tensorboard_target='gs://variational_shaker_models/tensorboard/assembly_based_jax_adi',
    encoder_model=encoder_model,
    conditioner=conditioner,
    decoder_model=decoder_model,
    optimizer=optimizer,
    compute_loss_fn=first_model.ComputeLoss,
    encoder_params=encoder_params,
    conditioner_params=conditioner_params,
    decoder_params=decoder_params,
    opt_state=opt_state,
    start_cpu_step=0)

if __name__ == "__main__":
  main()
