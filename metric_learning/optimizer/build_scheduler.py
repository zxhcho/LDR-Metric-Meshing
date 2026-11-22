from utils.registry import Register
import torch

SCHEDULER_REGISTRY = Register()

SCHEDULER_REGISTRY["StepLR"] = torch.optim.lr_scheduler.StepLR
SCHEDULER_REGISTRY["Exponential"] = torch.optim.lr_scheduler.ExponentialLR


def build_scheduler(config, optimizer):
    sche_type = config.scheduler.type
    sche_args = {}
    if sche_type == "StepLR":
        sche_args["step_size"] = config.scheduler.step_size
        sche_args["gamma"] = config.scheduler.gamma
    elif sche_type == "Exponential":
        sche_args["gamma"] = config.scheduler.gamma
    return SCHEDULER_REGISTRY[sche_type](optimizer, **sche_args)
