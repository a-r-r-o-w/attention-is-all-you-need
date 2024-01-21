import torch.nn as nn


def get_activation(name: str, **kwargs) -> nn.Module:
    if name == "relu":
        return nn.ReLU(**kwargs)
    elif name == "gelu":
        return nn.GELU(**kwargs)
    elif name == "silu" or name == "swish":
        return nn.SiLU(**kwargs)
    elif name == "leaky_relu":
        return nn.LeakyReLU(**kwargs)
    raise ValueError(f"{name} is not a supported activation")
