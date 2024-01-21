import torch
import torch.nn as nn

T = torch.Tensor


class PositionalEncoding(nn.Module):
    r"""Positional Encoding (Section 3.5 of paper).
    
    Args:
    """
    
    def __init__(
        self,
        embedding_size: int, # `d_model` in paper
        max_length: int = 10000,
    ) -> None:
        super().__init__()

        self.embedding_size = embedding_size
        self.max_length = max_length

        two_i = torch.arange(0, embedding_size, 2).float()
        numerator = torch.arange(0, max_length).float().unsqueeze(1)
        denominator = 10000.0 ** (two_i / embedding_size)

        pe = torch.zeros(max_length, embedding_size)
        pe[:, 0::2] = torch.sin(numerator / denominator)
        pe[:, 1::2] = torch.cos(numerator / denominator)
        
        # PE does not require gradients
        self.register_parameter("pe", torch.nn.Parameter(pe, requires_grad=False))
        self.pe: torch.nn.Parameter
    
    def forward(self, x: T) -> T:
        seq_length = x.size(1)
        return self.pe[:seq_length, :]
        