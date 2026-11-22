import os
import yaml
import torch

from abc import ABC, abstractmethod
from model.build_model import build_model
from model.build_model_hash import build_model_hash
from dataloader.build_dataloader import build_dataloader
from optimizer.build_optimizer import build_optimizer
from optimizer.build_scheduler import build_scheduler
from logger.build_logger import build_logger
from criterion.build_criterion import CalcCriterion
from utils.stat import get_git_revision_short_hash


class Trainer(ABC):
    def __init__(self, config) -> None:
        self.device = config.device

        self.config = config
        self.base_path = config.trainer.dataset.path
        self.model_hash = build_model_hash(config)

        # model save directory
        self.model_save_dir = os.path.join(
            self.base_path, "saved_models", self.model_hash)
        os.makedirs(self.model_save_dir, exist_ok=True)
        git_hash = get_git_revision_short_hash()
        config.hash_id = git_hash
        config_save_path = os.path.join(self.model_save_dir, "config.yaml")
        yaml.dump(config.__dict__, open(config_save_path, 'w'))

        dataset_type = config.trainer.dataset.type
        print(" ==> loading train dataset")
        self.train_loader = build_dataloader(
            config=config, mode="train", dataset_type=dataset_type,
            root_dir=os.path.join(self.base_path, "train"))
        print(" ==> loading evaluate dataset")
        self.evaluate_loader = build_dataloader(
            config=config, mode="evaluate", dataset_type=dataset_type,
            root_dir=os.path.join(self.base_path, "evaluate"))

        # load model
        model_type = config.model.type
        self.model = build_model(model_type, config).to(self.device)

        # load optimizer and scheduler
        self.optimizer = build_optimizer(config, self.model)
        self.scheduler = build_scheduler(config, self.optimizer)

        # load logger
        self.logger = build_logger(config, "train")

        # criterion
        self.criterion = CalcCriterion(config)

    def save_model(self, epoch):
        print(" ==> saving {} model".format(epoch))
        torch.save(self.model.state_dict(), os.path.join(
            self.model_save_dir, "model_{}.pth".format(epoch)))
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        os.makedirs(os.path.join(self.model_save_dir,
                    "checkpoint_{}".format(epoch)), exist_ok=True)
        torch.save(checkpoint, os.path.join(
            self.model_save_dir, "checkpoint_{}".format(epoch), "checkpoint.pth"))

    @abstractmethod
    def step(self, mode, epoch):
        raise NotImplementedError

    def train(self, total_epoch=100):
        for epoch in range(total_epoch + 1):
            self.model.train()
            self.optimizer.zero_grad()
            self.step(mode="train", epoch=epoch)
            self.scheduler.step()

            if epoch % self.config.epoch.evaluate_every_epoch == 0 or epoch == total_epoch+1:
                self.model.eval()
                self.step(mode="evaluate", epoch=epoch)

            if (epoch and epoch % self.config.epoch.save_every_epoch == 0 or epoch == total_epoch+1):
                self.save_model(epoch)
