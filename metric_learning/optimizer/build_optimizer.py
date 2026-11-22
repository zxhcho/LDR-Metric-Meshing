from utils.registry import Register
import torch

OPTIMIZER_REGISTRY = Register()

OPTIMIZER_REGISTRY["SGD"] = torch.optim.SGD
OPTIMIZER_REGISTRY["Adam"] = torch.optim.Adam
OPTIMIZER_REGISTRY["AdamW"] = torch.optim.AdamW


def build_optimizer(config, model):
    opt_type = config.optimizer.type
    opt_args = {}
    opt_args["lr"] = config.optimizer.lr
    if opt_type == "SGD":
        opt_args["momentum"] = config.optimizer.momentum
        opt_args["weight_decay"] = config.optimizer.weight_decay
    elif opt_type == "AdamW":
        opt_args["weight_decay"] = config.optimizer.weight_decay

    return OPTIMIZER_REGISTRY[opt_type](model.parameters(), **opt_args)
