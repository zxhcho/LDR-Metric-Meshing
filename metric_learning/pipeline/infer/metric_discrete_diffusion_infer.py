import torch
import os
import numpy as np

from collections import defaultdict
from tqdm import tqdm
from pipeline.infer.infer import Infer
from pipeline.infer.build_infer import INFER_REGISTRY
from vtk_io.vtk_io import write_vtk_with_attribute


@INFER_REGISTRY("DiscreteMetricDiffusion")
class DiscreteMetricDiffusionNetInfer(Infer):
    def __init__(self, config, mode) -> None:
        super().__init__(config, mode)

    def infer(self):
        data_loader = self.predict_loader
        mode = self.mode

        n_total = len(data_loader.dataset)
        with torch.no_grad():
            total_loss_dic = defaultdict(float)
            for i, data in tqdm(enumerate(data_loader), total=n_total):
                for key in data.keys():
                    data[key] = data[key].to(self.device)

                logits = self.model(data)

                if mode == "evaluate":
                    loss_dic = self.criterion(
                        pred=logits, epoch=0, **data)
                    for k, v in loss_dic.items():
                        total_loss_dic[k] += v.item() / n_total

                out_path = self.get_out_filename(
                    data_loader.dataset.file_names_list[i])

                verts = data["verts"].cpu().numpy()
                ev = data["ev"].cpu().numpy()

                metric = logits.cpu().numpy().reshape(-1) / self.config.model.edge_scale
                metric = np.abs(metric) + 1e-20
                if self.config.model.edge_type == "halfedge":
                    ev = np.concatenate([ev, ev], axis=0)
                write_vtk_with_attribute(
                    out_path, verts, ev, metric, "metric")

            if mode == "evaluate":
                self.logger(epoch=0, loss_dic=total_loss_dic, mode=mode)


@INFER_REGISTRY("MatrixMetricDiffusion")
class MatrixMetricDiffusionNetInfer(Infer):
    def __init__(self, config, mode) -> None:
        super().__init__(config, mode)

    def infer(self):
        data_loader = self.predict_loader
        mode = self.mode

        n_total = len(data_loader.dataset)
        with torch.no_grad():
            total_loss_dic = defaultdict(float)
            for i, data in tqdm(enumerate(data_loader), total=n_total):
                for key in data.keys():
                    data[key] = data[key].to(self.device)

                logits = self.model(data)

                if mode == "evaluate":
                    loss_dic = self.criterion(
                        pred=logits, epoch=0, **data)
                    for k, v in loss_dic.items():
                        total_loss_dic[k] += v.item() / n_total

                    out_path = self.get_out_filename(
                        data_loader.dataset.file_names_list[i])

                    verts = data["verts"].cpu().numpy()
                    faces = data["faces"].cpu().numpy()
                    gt_metric = data["metric"].cpu().numpy().reshape(-1, 3)
                    pred_metric = logits.cpu().numpy().reshape(-1, 3)
                    metric_diff = np.abs(pred_metric - gt_metric)
                    metric_diff_norm = np.linalg.norm(
                        metric_diff, axis=1, ord=2).reshape(-1)
                    gt_metric_norm = np.linalg.norm(
                        gt_metric, axis=1, ord=2).reshape(-1)
                    rel_error = metric_diff_norm / (gt_metric_norm + 1e-20)

                    write_vtk_with_attribute(
                        out_path, verts, faces, rel_error, "rel_error")

            if mode == "evaluate":
                self.logger(epoch=0, loss_dic=total_loss_dic, mode=mode)
