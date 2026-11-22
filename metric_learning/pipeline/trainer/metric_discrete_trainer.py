import torch
import os

from collections import defaultdict
from tqdm import tqdm
from contextlib import nullcontext
from pipeline.trainer.build_trainer import TRAINER_REGISTRY
from pipeline.trainer.trainer import Trainer
from dataloader.build_dataloader import build_dataloader


@TRAINER_REGISTRY("DiscreteMetricDiffusion")
class DiscreteMetricDiffusionNetTrainer(Trainer):
    def __init__(self, config) -> None:
        super().__init__(config)

        dataset_type = config.trainer.dataset.type
        print(" ==> loading predict dataset")
        self.predict_loader = build_dataloader(
            config=config, mode="evaluate", dataset_type=dataset_type,
            root_dir=os.path.join(self.base_path, "predict"))

    def step(self, mode, epoch):
        if mode == "train":
            dataloader = self.train_loader
        elif mode == "evaluate":
            dataloader = self.evaluate_loader
        elif mode == "predict":
            dataloader = self.predict_loader

        n_total = len(dataloader.dataset)
        with torch.no_grad() if mode != "train" else nullcontext():
            total_loss_dic = defaultdict(float)
            for data in tqdm(dataloader):
                for key in data.keys():
                    data[key] = data[key].to(self.device)

                logits = self.model(data)
                loss_dic = self.criterion(pred=logits, epoch=epoch, **data)
                loss = sum(loss_dic.values())

                if mode == "train":
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                with torch.no_grad():
                    for k, v in loss_dic.items():
                        total_loss_dic[k] += v.item() / n_total

            self.logger(epoch=epoch, loss_dic=total_loss_dic, mode=mode)

    def train(self, total_epoch=100):
        for epoch in range(total_epoch + 1):
            self.model.train()
            self.optimizer.zero_grad()
            self.step(mode="train", epoch=epoch)
            self.scheduler.step()

            if epoch % self.config.epoch.evaluate_every_epoch == 0 or epoch == total_epoch+1:
                self.model.eval()
                self.step(mode="evaluate", epoch=epoch)
                self.step(mode="predict", epoch=epoch)

            if (epoch and epoch % self.config.epoch.save_every_epoch == 0 or epoch == total_epoch+1):
                self.save_model(epoch)


@TRAINER_REGISTRY("MatrixMetricDiffusion")
class MatrixMetricDiffusionNetTrainer(Trainer):
    def __init__(self, config) -> None:
        super().__init__(config)

        dataset_type = config.trainer.dataset.type
        print(" ==> loading predict dataset")
        self.predict_loader = build_dataloader(
            config=config, mode="evaluate", dataset_type=dataset_type,
            root_dir=os.path.join(self.base_path, "predict"))

    def step(self, mode, epoch):
        if mode == "train":
            dataloader = self.train_loader
        elif mode == "evaluate":
            dataloader = self.evaluate_loader
        elif mode == "predict":
            dataloader = self.predict_loader

        n_total = len(dataloader.dataset)
        with torch.no_grad() if mode != "train" else nullcontext():
            total_loss_dic = defaultdict(float)
            for data in tqdm(dataloader):
                for key in data.keys():
                    data[key] = data[key].to(self.device)

                logits = self.model(data)
                loss_dic = self.criterion(pred=logits, epoch=epoch, **data)
                loss = sum(loss_dic.values())

                if mode == "train":
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                with torch.no_grad():
                    for k, v in loss_dic.items():
                        total_loss_dic[k] += v.item() / n_total

            self.logger(epoch=epoch, loss_dic=total_loss_dic, mode=mode)

    def train(self, total_epoch=100):
        for epoch in range(total_epoch + 1):
            self.model.train()
            self.optimizer.zero_grad()
            self.step(mode="train", epoch=epoch)
            self.scheduler.step()

            if epoch % self.config.epoch.evaluate_every_epoch == 0 or epoch == total_epoch+1:
                self.model.eval()
                self.step(mode="evaluate", epoch=epoch)
                self.step(mode="predict", epoch=epoch)

            if (epoch and epoch % self.config.epoch.save_every_epoch == 0 or epoch == total_epoch+1):
                self.save_model(epoch)
