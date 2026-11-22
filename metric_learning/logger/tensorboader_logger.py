from torch.utils.tensorboard import SummaryWriter
from abc import abstractmethod, ABC


class TensorboardLogger(ABC):
    def __init__(self, logger_path=None) -> None:
        if logger_path is not None:
            self.writer = SummaryWriter(logger_path)
        else:
            self.writer = None

    @abstractmethod
    def __call__(self, epoch, preds, gt, loss_dic, mode):
        raise NotImplementedError
