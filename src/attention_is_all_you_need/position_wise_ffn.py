import torch
import torch.nn as nn

T = torch.FloatTensor


def get_activation(name: str, **kwargs) -> nn.Module:
    if name == "relu" or name == "reglu":
        return nn.ReLU(**kwargs)
    elif name == "gelu" or name == "geglu":
        return nn.GELU(**kwargs)
    elif name == "silu" or name == "swish" or name == "swiglu":
        return nn.SiLU(**kwargs)
    elif name == "sigmoid" or name == "glu":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh(**kwargs)
    elif name == "elu":
        return nn.ELU(**kwargs)
    elif name == "leaky_relu":
        return nn.LeakyReLU(**kwargs)
    raise ValueError(f"{name} is not a supported activation")


def _is_glu_activation(activation: str) -> bool:
    return activation in ["glu", "reglu", "geglu", "swiglu"]


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
        dropout_rate: float = 0.1,
        use_bias: bool = True,
    ) -> None:
        super().__init__()

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_bias = use_bias

        self.in_proj = nn.Linear(in_out_dim, hidden_dim, bias=use_bias)
        self.out_proj = nn.Linear(hidden_dim, in_out_dim, bias=use_bias)
        self.gates = (
            nn.Linear(in_out_dim, hidden_dim, bias=use_bias)
            if _is_glu_activation(activation)
            else None
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.act = get_activation(activation)

    def forward(self, x: T) -> T:
        if self.gates is None:
            in_proj = self.in_proj(x)
            x = self.act(in_proj)
        else:
            in_proj = self.in_proj(x)
            gate = self.gates(x)
            x = self.act(gate) * in_proj

        x = self.dropout(x)
        x = self.out_proj(x)

        return x
