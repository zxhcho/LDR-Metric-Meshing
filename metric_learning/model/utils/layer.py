import torch
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self, input_dim, output_dim, with_bn=False, with_ac=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.with_bn = with_bn
        self.with_ac = with_ac

        if with_bn:
            self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        """
        x: (*, C)
        """
        x = self.fc(x)
        if self.with_bn:
            dim_x = len(x.shape)
            if dim_x == 2:
                x = self.bn(x)
            elif dim_x == 3:
                x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        if self.with_ac:
            x = F.relu(x, inplace=True)
        return x


class Conv1D(nn.Module):
    def __init__(self, input_dim, output_dim, with_bn=False, with_ac=True):
        super(Conv1D, self).__init__()
        self.with_bn = with_bn
        self.with_ac = with_ac
        self.conv = nn.Conv1d(input_dim, output_dim, 1)
        if with_bn:
            self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        """
        x: (B, C, N)
        """
        x = self.fc(x)
        if self.with_bn:
            x = self.bn(x)
        if self.with_ac:
            x = F.relu(x, inplace=True)
        return x


class Aggregator(nn.Module):
    def __init__(self, oper):
        super().__init__()
        assert oper in ('mean', 'sum', 'max')
        self.oper = oper

    def forward(self, x, dim=2):
        if self.oper == 'mean':
            return x.mean(dim=dim, keepdim=False)
        elif self.oper == 'sum':
            return x.sum(dim=dim, keepdim=False)
        elif self.oper == 'max':
            ret = x.max(dim=dim, keepdim=False)[0]
            return ret


class MLP(nn.Module):
    def __init__(self,
                 sizes=[],
                 last_ac=False) -> None:
        super().__init__()
        assert (len(sizes) >= 2)

        self.reg_layers = nn.ModuleList()
        for i in range(len(sizes)-2):
            self.reg_layers.append(FC(sizes[i], sizes[i+1], with_ac=True))
        self.reg_layers.append(FC(sizes[-2], sizes[-1], with_ac=last_ac))

    def forward(self, x):
        for layer in self.reg_layers:
            x = layer(x)
        return x


class PileConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PileConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_in = FC(input_dim, input_dim*2)
        self.conv_mid = FC(input_dim*2, input_dim//2)
        self.conv_out = FC(input_dim + input_dim//2, output_dim)

    def forward(self, x, num_out, dist_w=None):
        """
        args:
            x: (B, N, C)
            B: patch number
            N: patch size
            C: channel
            dist_w: (B, N, 1)
        rtn:
            x: (B, num_out, out_dim)
        """
        B, N, C = x.shape

        if dist_w is None:
            y = self.conv_in(x)
        else:
            y = self.conv_in(x * dist_w[:, :N, :])        # (B, N, C*2)

        feat_g = torch.max(y, dim=2, keepdim=True)[0]   # (B, 1, C*2)
        feat_g = self.conv_mid(feat_g)              # (B, 1, C/2)

        x = torch.cat([x[:, :num_out, :],
                       feat_g.repeat(1, num_out, 1),
                       ], dim=1)  # (B, num_out, C/2)

        x = self.conv_out(x)  # (B, num_out, out_dim)
        return x


class PatchAttention(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim) -> None:
        super().__init__()
        self.conv_q = FC(in_dim, out_dim)
        self.conv_v = FC(in_dim, out_dim)

    def forward(self, feat):
        """
        args:
            feat: (B, N, C)
        rtn:
            feat: (B, C)
        """
        query = self.conv_q(feat)  # (B, N, C)
        value = self.conv_v(feat)  # (B, N, C)
        attn = torch.softmax(query, dim=1).max(
            dim=2, keepdim=True)[0]  # (B, N, 1)

        # (B, C, N) * (B, N, 1) = (B, C, 1) -> (B, C)
        feat = torch.matmul(value.transpose(1, 2), attn).squeeze()
        return feat
