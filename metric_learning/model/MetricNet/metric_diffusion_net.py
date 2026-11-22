import torch
import torch.nn as nn


from model.diffusion_net import DiffusionNet
from model.utils.layer import MLP


class DiscreteEdgeMetricDiffusionNet(nn.Module):
    def __init__(self,
                 config,
                 in_channels) -> None:
        super().__init__()

        feat_dim = config.model.diffusion_net.C_width
        num_classes = 1

        self.diffusion_net = DiffusionNet(
            C_in=in_channels,
            C_out=num_classes,
            N_block=config.model.diffusion_net.N_block,
            C_width=feat_dim,
            outputs_at="vertices",
            use_reg_head=False)

        self.last_lin = MLP(
            sizes=[feat_dim*2, 512, 256, 128, 1], last_ac=False)

    def forward(self, data_dict):

        v_feat = self.diffusion_net(**data_dict)  # [nv, c]
        ev = data_dict["ev"]  # [ne, 2]
        edge_feat_diff = torch.square(
            v_feat[ev[:, 0], :] - v_feat[ev[:, 1], :])   # [ne, c]
        edge_feat_sum = torch.square(
            v_feat[ev[:, 0], :] + v_feat[ev[:, 1], :])   # [ne, c]
        edge_feat = torch.cat([edge_feat_sum, edge_feat_diff], dim=1)
        edge_len = self.last_lin(edge_feat)
        edge_len_square = torch.square(edge_len)

        return edge_len_square  # [ne, 1]


class DiscreteHalfedgeMetricDiffusionNet(nn.Module):
    def __init__(self,
                 config,
                 in_channels) -> None:
        super().__init__()

        feat_dim = config.model.diffusion_net.C_width
        num_classes = 1

        self.diffusion_net = DiffusionNet(
            C_in=in_channels,
            C_out=num_classes,
            N_block=config.model.diffusion_net.N_block,
            C_width=feat_dim,
            outputs_at="vertices",
            use_reg_head=False)

        self.last_lin = MLP(
            sizes=[feat_dim, 512, 256, 128, 1], last_ac=False)

    def forward(self, data_dict):

        v_feat = self.diffusion_net(**data_dict)  # [nv, c]
        ev = data_dict["ev"]  # [ne, 2]
        edge_feat = torch.square(
            v_feat[ev[:, 0], :] - v_feat[ev[:, 1], :])   # [ne, c]
        edge_feat = torch.cat([edge_feat, -edge_feat], dim=0)  # [2ne, c]
        edge_len = self.last_lin(edge_feat)
        edge_len_square = torch.square(edge_len)

        return edge_len_square  # [2ne, 1]


class MatrixMetricDiffusionNet(nn.Module):
    def __init__(self,
                 config,
                 in_channels) -> None:
        super().__init__()

        feat_dim = config.model.diffusion_net.C_width
        num_classes = 3

        self.diffusion_net = DiffusionNet(
            C_in=in_channels,
            C_out=num_classes,
            N_block=config.model.diffusion_net.N_block,
            C_width=feat_dim,
            outputs_at="vertices",
            use_reg_head=False)

        self.last_lin = MLP(
            sizes=[feat_dim, 512, 256, 128, num_classes], last_ac=False)

    def forward(self, data_dict):

        v_feat = self.diffusion_net(**data_dict)  # [nv, c]
        metric = self.last_lin(v_feat)  # [nv, 3]

        return metric  # [nv, 3]
