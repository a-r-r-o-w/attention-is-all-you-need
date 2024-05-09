import torch
import torch.nn as nn

from .utils import get_activation

T = torch.FloatTensor


class PositionwiseFeedForward(nn.Module):
    r"""Position-wise Feed-forward Network (section 3.3 in paper).

    Args:
        in_out_dim (`int`):
            The dimension of the input and output vectors.
        hidden_dim (`int`):
            The dimension of the hidden layer.
        activation (`str`, optional):
            The activation function to use. Defaults to `"relu"`.
        use_bias_1 (`bool`, optional):
            Whether to use bias in the first linear layer. Defaults to `True`.
        use_bias_2 (`bool`, optional):
            Whether to use bias in the second linear layer. Defaults to `True`.
    """

    def __init__(
        self,
        in_out_dim: int,
        hidden_dim: int,
        activation: str = "relu",
        use_bias_1: bool = True,
        use_bias_2: bool = True,
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_out_dim, hidden_dim, bias=use_bias_1)
        self.linear_2 = nn.Linear(hidden_dim, in_out_dim, bias=use_bias_2)
        self.activation = get_activation(activation)

    def forward(self, x: T) -> T:
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        return x
