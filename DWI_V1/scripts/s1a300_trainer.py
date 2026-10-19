"""S1a promotion trainer: resume the 100-epoch checkpoint and stop at epoch 300."""

from recall_trainers import nnUNetTrainer_S1a
from screen100_trainers import Screen100Mixin


class nnUNetTrainer_S1a_Promote300(Screen100Mixin, nnUNetTrainer_S1a):
    """Same S1a configuration and 1000-epoch PolyLR trajectory, total epoch 300."""

    screening_epochs = 300

