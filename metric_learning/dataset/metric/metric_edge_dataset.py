from abc import abstractmethod
from tqdm import tqdm
from dataset.preprocess.calc_op_cache import load_op_cache
from dataset.base.base_dataset import BaseDataset
from dataset.build_dataset import DATASET_REGISTRY

import os
import torch
import numpy as np


@DATASET_REGISTRY("DiffusionMetricEdgeConsistentMesh")
class DiffusionMetricEdgeConsistentMeshDataset(BaseDataset):
    """
        The background mesh is the same, only the signals/field on it are different.
    """

    def __init__(self, config, root_dir, mode="train") -> None:
        super(DiffusionMetricEdgeConsistentMeshDataset,
              self).__init__(root_dir, mode)

        self.feature_types = config.model.feature_types
        self.edge_type = config.model.edge_type
        self.edge_scale = config.model.edge_scale

        self.data_list = []

        op_cache_dir = os.path.join(root_dir, "../op_cache")
        op_cache_file = os.listdir(op_cache_dir)[0]
        self.op_cache_file = os.path.join(op_cache_dir, op_cache_file)
        self.op_cache_data = {}

        # feat_dir = os.path.join(root_dir, "../feature", "hks")
        # feat_file = os.listdir(feat_dir)[0]
        # self.feat_file = os.path.join(feat_dir, feat_file)
        # self.feat_data = {}

        self.file_names_list = self.load_filenames()
        self.load_file(self.file_names_list)

    def load_filenames(self):
        edge_dir = self.root_dir
        file_names_list = []
        for fname in os.listdir(edge_dir):
            if fname.endswith(".npz"):
                file_names_list.append(os.path.join(edge_dir, fname))
        return file_names_list

    def load_file(self, file_names_list):
        self.op_cache_data = load_op_cache(self.op_cache_file, False)
        # self.feat_data = np.load(self.feat_file)
        for i in tqdm(range(len(file_names_list))):
            filepath = file_names_list[i]
            data = {}
            npz_data = np.load(filepath)
            data["ev"] = torch.from_numpy(
                npz_data["edges"]).to(dtype=torch.long)
            data["feature"] = torch.from_numpy(
                npz_data["field"].reshape(-1, 1)).to(dtype=torch.float32)
            if self.edge_type == "edge":
                data["length"] = torch.from_numpy(
                    npz_data["length"]).to(dtype=torch.float32).flatten()
            elif self.edge_type == "halfedge":
                # flatten the length in column major order
                data["length"] = torch.from_numpy(
                    npz_data["length"]).to(dtype=torch.float32).transpose(1, 0).flatten()
            data['length'] *= self.edge_scale

            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        augumentation/rotation should be implimented here
        """
        data = self.data_list[idx]
        data.update(self.op_cache_data)
        # data["feature"] = torch.cat(
        #     [data["feature"], data["field"]], dim=1)

        return data


@DATASET_REGISTRY("DiffusionMetricEdge")
class DiffusionMetricEdgeDataset(BaseDataset):
    """
        The background mesh is the same, only the signals/field on it are different.
    """

    def __init__(self, config, root_dir, mode="train") -> None:
        super(DiffusionMetricEdgeDataset,
              self).__init__(root_dir, mode)

        self.feature_types = config.model.feature_types
        self.edge_type = config.model.edge_type
        self.edge_scale = config.model.edge_scale

        self.data_list = []

        self.edge_dir = os.path.join(root_dir, self.edge_type)
        self.op_cache_dir = os.path.join(root_dir, "op_cache")

        self.file_names_list = self.load_filenames()
        self.load_file(self.file_names_list)

    def load_filenames(self):
        edge_dir = os.path.join(self.root_dir, self.edge_type)
        file_names_list = []
        for fname in os.listdir(edge_dir):
            if fname.endswith(".npz"):
                file_names_list.append(os.path.join(edge_dir, fname))
        return file_names_list

    def load_file(self, file_names_list):
        # self.op_cache_data = load_op_cache(self.op_cache_file, False)
        for i in tqdm(range(len(file_names_list))):
            edge_path = file_names_list[i]
            op_cache_path = edge_path.replace(self.edge_dir, self.op_cache_dir)
            data = load_op_cache(op_cache_path, False)
            npz_data = np.load(edge_path)
            data["ev"] = torch.from_numpy(
                npz_data["edges"]).to(dtype=torch.long)
            data["feature"] = torch.from_numpy(
                npz_data["field"].reshape(-1, 1)).to(dtype=torch.float32)
            if self.edge_type == "edge":
                data["length"] = torch.from_numpy(
                    npz_data["length"]).to(dtype=torch.float32).flatten()
            elif self.edge_type == "halfedge":
                # flatten the length in column major order
                data["length"] = torch.from_numpy(
                    npz_data["length"]).to(dtype=torch.float32).transpose(1, 0).flatten()
            data['length'] *= self.edge_scale

            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        augumentation/rotation should be implimented here
        """
        data = self.data_list[idx]

        return data
