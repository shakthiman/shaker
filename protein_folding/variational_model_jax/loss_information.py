import collections

LossInformation = collections.namedtuple(
    'LossInformation', ['loss', 'loss_beta_1', 'logpx_z', 'logpz', 'logqz_x', 'diff_mae'])
