from protein_folding.variational_model_jax import loss_information
from protein_folding.variational_model_jax.models import shared_modules
from protein_folding.variational_model_jax.models import shared_utils

from flax import linen as nn
from jax import numpy as jnp
from jax import random
from jax import scipy as jscipy

import typing

class EncoderModule(nn.Module):
  initial_gamma_value: float

  def _log_sigma2(self, gamma):
    return nn.log_sigmoid(-1*gamma)

  def _sigma2(self, gamma):
    return nn.sigmoid(-1*gamma)

  def _alpha(self, gamma):
    return jnp.sqrt(1 - self._sigma2(gamma))

  def setup(self):
    self._gamma = self.param('gamma',
                            nn.initializers.constant(self.initial_gamma_value),
                             [])
  def __call__(self, training_data):
    a = self._alpha(self._gamma)
    logvar = self._log_sigma2(self._gamma)
    return (a*training_data['normalized_coordinates'], logvar)


class DecoderModule(nn.Module):
  transformers: typing.List[shared_modules.EfficientTransformerUnit]
  initial_scale_value: float

  def setup(self):
    self.final_layer = nn.Dense(3)
    self.scale = self.param('scale',
                            nn.initializers.constant(self.initial_scale_value),
                            [])

  def log_prob_x(self, conditioning, latent_embeddings, mask,
                 normalized_coordinates, deterministic):
    mean_val = self.mean_prediction(conditioning, latent_embeddings, mask,
                                    deterministic=deterministic)
    return (mean_val,
            jscipy.stats.norm.logpdf(
                normalized_coordinates, mean_val, self.scale))

  def mean_prediction(self, conditioning, latent_embeddings, mask,
                      deterministic):
    o = conditioning
    for t in self.transformers:
      o = t(jnp.concatenate([o, latent_embeddings], axis=-1), mask,
            deterministic=deterministic)
    return self.final_layer(
        jnp.concatenate([o, latent_embeddings], axis=-1))

class ConditionerModule(nn.Module):
  amino_acid_embedding_dims: int
  max_atom_indx: float
  residue_lookup_size: int
  residue_embedding_dims: int
  atom_lookup_size: int
  atom_embedding_dims: int

  def _atom_indices_embedding(self,
                      atom_indices):
    half_dim = self.amino_acid_embedding_dims // 2
    pemb = jnp.log(self.max_atom_indx) / (half_dim - 1)
    pemb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -pemb)
    pemb = (jnp.expand_dims(atom_indices, -1) *
            jnp.expand_dims(jnp.expand_dims(pemb, 0), 0))
    pemb = jnp.concatenate([jnp.sin(pemb), jnp.cos(pemb)], axis=-1)
    return pemb

  def setup(self):
    self.residue_embeddings = nn.Embed(
        self.residue_lookup_size, self.residue_embedding_dims)
    self.atom_names_embeddings = nn.Embed(
        self.atom_lookup_size, self.atom_embedding_dims)

  def __call__(self, training_data):
    return jnp.concatenate([
        jnp.expand_dims(training_data['peptide_indices'], -1),
        self._atom_indices_embedding(training_data['atom_indices']),
        self.residue_embeddings(training_data['residue_names']),
        self.atom_names_embeddings(training_data['atom_names'])], axis=-1)

def ComputeLoss(random_key,
                encoder_model,
                decoder_model,
                conditioner_model,
                encoder_params,
                decoder_params,
                conditioner_params,
                training_data):
  # See https://www.tensorflow.org/tutorials/generative/cvae#define_the_loss_function_and_the_optimizer
  # for definition of the loss functions
  conditioning = conditioner_model.apply(conditioner_params, training_data)
  mean_z, logvar_z = encoder_model.apply(encoder_params, training_data)
  eps = random.normal(random_key, shape=jnp.shape(mean_z))
  z = eps * jnp.exp(logvar_z * 0.5) + mean_z

  mask = shared_utils.Mask(training_data)
  mean_val, log_prob_x_z = decoder_model.apply(
      decoder_params, conditioning, z, mask,
      training_data['normalized_coordinates'],
      method=DecoderModule.log_prob_x)
  log_prob_z = shared_utils.LogNormalPdf(z, 0, 0)
  log_prob_z_x = shared_utils.LogNormalPdf(z, mean_z, logvar_z)

  log_prob_x_z = jnp.mean(
      jnp.sum(log_prob_x_z*jnp.expand_dims(mask, -1), axis=[1,2]), axis=0)
  log_prob_z = jnp.mean(
      jnp.sum(log_prob_z*jnp.expand_dims(mask, -1), axis=[1,2]), axis=0)
  log_prob_z_x = jnp.mean(
      jnp.sum(log_prob_z_x*jnp.expand_dims(mask, -1), axis=[1,2]), axis=0)

  diff_mae = shared_utils.DiffMAE(mean_val=mean_val,
                                  training_data=training_data,
                                  mask=mask)

  return loss_information.LossInformation(
      loss=-1*(log_prob_x_z + log_prob_z - log_prob_z_x),
      loss_beta_1=-1*(log_prob_x_z + log_prob_z - log_prob_z_x),
      logpx_z= log_prob_x_z,
      logpz = log_prob_z,
      logqz_x=log_prob_z_x,
      diff_mae=diff_mae)

def GetModels(batch_size, input_length, num_blocks, pdb_vocab):
  #Instantiate the Encoder, Decoder, and Conditioner
  encoder_model = EncoderModule(6.0)
  conditioner = ConditionerModule(
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
  decoder_model = DecoderModule(
      transformers=[shared_modules.EfficientTransformerUnit(
        num_blocks=num_blocks,
        input_length=input_length,
        batch_size=batch_size,
        local_transformer=nn.MultiHeadAttention(num_heads=5, qkv_features=10),
        global_transformer=nn.MultiHeadAttention(num_heads=5, qkv_features=10),
        feedforward_network=shared_modules.DNN(
            dnn_layers=[nn.Dense(100) for i in range(5)],
            activation_function = nn.gelu
        ),
        final_layer=nn.Dense(num_conditioner_features)
    ) for i in range(10)],
    initial_scale_value=1)
  return (encoder_model, conditioner, decoder_model)

def Init(random_key, encoder, decoder, conditioner, batch_size, input_length):
  random_key, variables_key = random.split(random_key, 2)
  random_key, params_key = random.split(random_key, 2)

  (peptide_indices_key,
   atom_indices_key,
   residue_names_key,
   atom_names_key,
   conditioning_key,
   z_key,
   mask_key,
   normalized_coordinates_key) = random.split(variables_key, 8)
  num_conditioner_features = (
      1 
      + conditioner.amino_acid_embedding_dims
      + conditioner.residue_embedding_dims
      + conditioner.atom_embedding_dims)
  conditioning = random.normal(conditioning_key,
                               (batch_size, input_length, num_conditioner_features))
  normalized_coordinates = random.normal(normalized_coordinates_key, (batch_size, input_length, 3))
  mask = random.bernoulli(mask_key, shape=(batch_size, input_length))
  z = random.normal(z_key, (batch_size,input_length, 3))
  peptide_indices = random.randint(peptide_indices_key, (batch_size, input_length), 0, 10)
  atom_indices = random.randint(atom_indices_key, (batch_size, input_length), 0, 10)
  residue_names = random.randint(residue_names_key, (batch_size, input_length), 0, 10)
  atom_names = random.randint(atom_names_key, (batch_size, input_length), 0, 10)

  encoder_key, conditioner_key, decoder_key = random.split(params_key, 3)
  encoder_params = encoder.init(encoder_key, normalized_coordinates)
  decoder_params = decoder.init(
      decoder_key, conditioning, z, mask, normalized_coordinates,
      method=DecoderModule.log_prob_x)
  conditioner_params = conditioner.init(
      conditioner_key, peptide_indices, atom_indices, residue_names, atom_names)
  return (encoder_params, conditioner_params, decoder_params)
