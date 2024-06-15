from jax import numpy as jnp
def Mask(training_data):
  return jnp.any(training_data['normalized_coordinates']!=0, axis=-1)

def LogNormalPdf(sample, mean, logvar):
  log2pi = jnp.log(2*jnp.pi)
  return -0.5*((sample-mean)**2. * jnp.exp(-logvar) + logvar + log2pi)

def DiffMAE(mean_val, training_data, mask):
  return jnp.sum(
      jnp.absolute(
        mean_val-training_data['normalized_coordinates'])
      * jnp.expand_dims(mask, -1))/jnp.sum(mask)
