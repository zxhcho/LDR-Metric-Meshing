from logger.tensorboader_logger import TensorboardLogger
from model.build_model_hash import build_model_hash
from logger.build_logger import LOGGER_REGISTRY

import torch
import os


@LOGGER_REGISTRY("Regression")
class RegressionLogger(TensorboardLogger):
    def __init__(self, config, mode) -> None:
        model_hash = build_model_hash(config)
        logger_path = os.path.join(
            config.logger.path, model_hash) if mode == "train" else None
        super(RegressionLogger, self).__init__(logger_path)

    def __call__(self, epoch, loss_dic, mode):

        loss = 0.0 if loss_dic is None else sum(
            loss_dic.values()) / len(loss_dic)
        if self.writer is not None:
            self.writer.add_scalar(f'Loss/{mode}', loss, epoch)
            for key, value in loss_dic.items():
                self.writer.add_scalar(f'Loss/{key}/{mode}', value, epoch)

        print(
            "Epoch {} - [{}], Loss: {:.4}".format(epoch, mode, loss))
