import collections

LossInformation = collections.namedtuple(
    'LossInformation', [
      'loss', 'loss_beta_1', 'logpx_z', 'logpz', 'logqz_x', 'diff_mae',
      'loss_alpha_carbon_clash', 'num_hard_clashes', 'num_soft_clashes'])

def CreateLossInformation(
    loss, loss_beta_1, logpx_z, logpz, logqz_x, diff_mae,
    loss_alpha_carbon_clash=0, num_hard_clashes=0, num_soft_clashes=0):
  return LossInformation(
      loss=loss,
      loss_beta_1=loss_beta_1,
      logpx_z=logpx_z,
      logpz=logpz,
      logqz_x=logqz_x,
      diff_mae=diff_mae,
      loss_alpha_carbon_clash=loss_alpha_carbon_clash,
      num_hard_clashes=num_hard_clashes,
      num_soft_clashes=num_soft_clashes)
