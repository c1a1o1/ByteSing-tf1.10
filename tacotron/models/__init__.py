from .tacotron import Tacotron
from .duration import Duration


def create_model(name, hparams):
  if name == 'Tacotron':
    return Tacotron(hparams)
  if name == 'Duration':
    return Duration(hparams)
  else:
    raise Exception('Unknown model: ' + name)
