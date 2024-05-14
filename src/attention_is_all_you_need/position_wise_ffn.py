import torch
import torch.nn as nn

T = torch.FloatTensor


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


class PositionwiseFeedForward(nn.Module):
    r"""Position-wise Feed-forward Network (section 3.3 in paper).

    Args:
        in_out_dim (`int`):
            The dimension of the input and output vectors.
        hidden_dim (`int`):
            The dimension of the hidden layer.
        activation (`str`, optional):
            The activation function to use. Defaults to `"relu"`.
        use_bias (`bool`, optional):
            Whether to use bias in the linear layers. Defaults to `True`.
    """

    def __init__(
        self,
        in_out_dim: int,
        hidden_dim: int,
        activation: str = "relu",
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_out_dim, hidden_dim, bias=use_bias)
        self.linear_2 = nn.Linear(hidden_dim, in_out_dim, bias=use_bias)
        self.activation = get_activation(activation)

    def forward(self, x: T) -> T:
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        return x
