import torch
import torch.nn as nn

from .utils import get_activation

T = torch.FloatTensor


class PositionwiseFeedForward(nn.Module):
    r"""Position-wise Feed-forward Network (section 3.3 in paper).

    Args:
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
