import torch
import torch.nn as nn


def get_summary(module: nn.Module):
    trainable_parameters = sum([p.numel() for p in module.parameters() if p.requires_grad])
    untrainable_parameters = sum([p.numel() for p in module.parameters() if not p.requires_grad])
    
    dtype = next(module.parameters()).dtype
    size = 2 if dtype == torch.float16 else 4
    memory = size * (trainable_parameters + untrainable_parameters) / 1024 / 1024
    
    return {
        "trainable": trainable_parameters,
        "untrainable": untrainable_parameters,
        "memory": memory,
    }


def initialize_weights(module: nn.Module, method: str, **kwargs) -> None:
    if method == "kaiming_uniform":
        m = nn.init.kaiming_uniform_
    elif method == "kaiming_normal":
        m = nn.init.kaiming_normal_
    elif method == "xavier_uniform":
        m = nn.init.xavier_uniform_
    elif method == "xavier_normal":
        m = nn.init.xavier_normal_
    elif method == "uniform":
        m = nn.init.uniform_
    elif method == "normal":
        m = nn.init.normal_
    
    def init(x):
        if hasattr(x, "weight") and x.weight.dim() > 1:
            m(x.weight.data, **kwargs)
    
    module.apply(init)
