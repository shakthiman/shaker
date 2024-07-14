import collections
from protein_folding.variational_model_jax.models import auxilliary_losses

LossInformation = collections.namedtuple(
    'LossInformation', [
      'loss', 'loss_beta_1', 'logpx_z', 'logpz', 'logqz_x', 'diff_mae',
      'loss_alpha_carbon_clash', 'num_hard_clashes', 'num_soft_clashes',
      'clash_sum_squares', 'loss_dihedral_loss', 'dihedral_loss'])

def CreateLossInformation(
    loss, loss_beta_1, logpx_z, logpz, logqz_x, diff_mae,
    loss_alpha_carbon_clash=0, num_hard_clashes=0, num_soft_clashes=0,
    clash_sum_squares=0, loss_dihedral_loss=0.0, dihedral_loss=auxilliary_losses.DihedralLoss(
        total_phi_error=0.0,
        total_psi_error=0.0,
        total_omega_error=0.0,
        average_phi_error=0.0,
        average_psi_error=0.0,
        average_omega_error=0.0)):
  return LossInformation(
      loss=loss,
      loss_beta_1=loss_beta_1,
      logpx_z=logpx_z,
      logpz=logpz,
      logqz_x=logqz_x,
      diff_mae=diff_mae,
      loss_alpha_carbon_clash=loss_alpha_carbon_clash,
      num_hard_clashes=num_hard_clashes,
      num_soft_clashes=num_soft_clashes,
      clash_sum_squares=clash_sum_squares,
      loss_dihedral_loss=loss_dihedral_loss,
      dihedral_loss=dihedral_loss)
