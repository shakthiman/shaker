from protein_folding.variational_model_jax.models import first_model
from protein_folding.variational_model_jax.models import shared_modules
from protein_folding.variational_model_jax.models import shared_utils

from flax import linen as nn

from jax import numpy as jnp
from jax import random

class VAE(nn.Module):
  encoder_model: first_model.EncoderModule
  decoder_model: first_model.DecoderModule
  conditioner_model: first_model.ConditionerModule

  def compute_model_loss(
      self, random_key, training_data):
    conditioning = self.conditioner_model(training_data)
    mean_z, logvar_z = self.encoder_model(training_data)
    eps = random.normal(random_key, shape=jnp.shape(mean_z))
    z = eps * jnp.exp(logvar_z * 0.5) + mean_z

    mask = shared_utils.Mask(training_data)
    mean_val, log_prob_x_z = self.decoder_model.log_prob_x_z(
        conditioning, z, mask, training_data['normalized_coordinates'])
    log_prob_z = shared_utils.LogNormalPdf(z, 0, 0)
    log_prob_z_x = shared_utils.LogNormalPdf(z, mean_z, logvar_z)

    log_prob_x_z = jnp.mean(
        jnp.sum(log_prob_x_z*jnp.expand_dims(mask, -1), axis=[1,2]), axis=0)
    log_prob_z = jnp.mean(
        jnp.sum(log_prob_z*jnp.expand_dims(mask, -1), axis=[1,2]), axis=0)
    log_prob_z_x = jnp.mean(
        jnp.sum(log_prob_z_x*jnp.expand_dims(mask, -1), axis=[1,2]), axis=0)
    return loss_information.LossInformation(
            loss=-1*(log_prob_x_z + log_prob_z - log_prob_z_x),
            loss_beta_1=-1*(log_prob_x_z + log_prob_z - log_prob_z_x),
            logpx_z= log_prob_x_z,
            logpz = log_prob_z,
            logqz_x=log_prob_z_x,
            diff_mae=diff_mae)

def GetModel(batch_size, input_length, num_blocks, pdb_vocab):
  #Instantiate the Encoder, Decoder, and Conditioner
  encoder_model = first_model.EncoderModule(6.0)
  conditioner = first_model.ConditionerModule(
      amino_acid_embedding_dims=20,
      max_atom_indx=float(input_length),
      residue_lookup_size=pdb_vocab.ResidueLookupSize(),
      residue_embedding_dims=3,
      atom_lookup_size=pdb_vocab.AtomLookupSize(),
      atom_embedding_dims=3)
  num_conditioner_features = (
      1 
      + conditioner.amino_acid_embedding_dims
      + conditioner.residue_embedding_dims
      + conditioner.atom_embedding_dims)
  decoder_model = first_model.DecoderModule(
      transformers=[shared_modules.EfficientTransformerUnit(
        num_blocks=num_blocks,
        input_length=input_length,
        batch_size=batch_size,
        local_transformer=nn.MultiHeadAttention(num_heads=100, qkv_features=num_conditioner_features),
        global_transformer=nn.MultiHeadAttention(num_heads=100, qkv_features=num_conditioner_features),
        feedforward_network=shared_modules.DNN(
            dnn_layers=[nn.Dense(100) for i in range(5)],
            activation_function = nn.gelu
        ),
        final_layer=nn.Dense(num_conditioner_features),
        dropout_fraction=0.1
    ) for i in range(10)],
    initial_scale_value=1)
  return VAE(
      encoder_model=encoder_model,
      conditioner_model=conditioner,
      decoder_model=decoder_model)

def Init(random_key, vae, batch_size, input_length):
  random_key, variables_key = random.split(random_key, 2)
  random_key, params_key = random.split(random_key, 2)

  (peptide_indices_key,
   atom_indices_key,
   residue_names_key,
   atom_names_key,
   normalized_coordinates_key) = random.split(variables_key, 5)

  peptide_indices = random.randint(peptide_indices_key, (batch_size, input_length), 0, 10)
  atom_indices = random.randint(atom_indices_key, (batch_size, input_length), 0, 10)
  residue_names = random.randint(residue_names_key, (batch_size, input_length), 0, 10)
  atom_names = random.randint(atom_names_key, (batch_size, input_length), 0, 10)
  normalized_coordinates = random.normal(normalized_coordinates_key, (batch_size, input_length, 3))

  return vae.init(
      params_key, {
        'peptide_indices': peptide_indices,
        'atom_indices': atom_indices,
        'residue_names': residue_names,
        'atom_names': atom_names,
        'normalized_coordinates': normalized_coordinates
        },
      method=VAE.compute_model_loss)
