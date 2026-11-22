import os
import torch

from abc import ABC, abstractmethod
from model.build_model import build_model
from model.build_model_hash import build_model_hash
from dataloader.build_dataloader import build_dataloader
from logger.build_logger import build_logger
from visualizer.build_visualizer import build_visualizer
from criterion.build_criterion import CalcCriterion


class Infer(ABC):
    def __init__(self, config, mode) -> None:
        self.device = config.device

        self.config = config
        self.mode = mode
        self.data_base_path = config.infer.dataset.path
        self.model_hash = build_model_hash(config)

        # predict save directory
        self.pred_save_dir = os.path.join(
            self.data_base_path+"_out", self.model_hash, f"model_{config.infer.model.epoch}.pth")
        os.makedirs(self.pred_save_dir, exist_ok=True)

        # model save directory
        self.model_path = os.path.join(
            config.infer.model.path, self.model_hash, f"model_{config.infer.model.epoch}.pth")
        if not os.path.exists(self.model_path):
            raise ValueError(
                f"model path {self.model_path} does not exist")

        # load model
        model_type = config.model.type
        self.model = build_model(model_type, config).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.to(self.device)
        self.model.eval()

        # load dataset
        dataset_type = config.infer.dataset.type
        self.predict_loader = build_dataloader(
            config=config, mode=mode, dataset_type=dataset_type,
            root_dir=self.data_base_path)

        # load logger
        self.logger = build_logger(config, "predict")

        self.visualizer = build_visualizer(config)

        # criterion
        self.criterion = CalcCriterion(config)

    def get_out_filename(self, in_filename):
        suffix = os.path.splitext(in_filename)[1]
        out_path = in_filename.replace(
            self.data_base_path, self.pred_save_dir).replace(suffix, ".vtk")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        return out_path

    @abstractmethod
    def infer(self):
        raise NotImplementedError
