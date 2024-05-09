import torch
import torch.nn as nn

T = torch.FloatTensor


class PositionalEncoding(nn.Module):
    r"""Positional Encoding (Section 3.5 of paper).

    Args:
        embedding_dim (`int`):
            The dimension of the embedding space (d_model in paper).
        max_length (`int`, optional):
            The maximum length of a sequence. Defaults to `10000`.
    """

    def __init__(
        self,
        embedding_dim: int,  # `d_model` in paper
        max_length: int = 10000,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_length = max_length

        two_i = torch.arange(0, embedding_dim, 2, dtype=torch.float32)
        numerator = torch.arange(0, max_length, dtype=torch.float32).unsqueeze(1)
        denominator = 10000.0 ** (two_i / embedding_dim)

        self.pe = torch.zeros(max_length, embedding_dim)
        self.pe[:, 0::2] = torch.sin(numerator / denominator)
        self.pe[:, 1::2] = torch.cos(numerator / denominator)

    def forward(self, x: T) -> T:
        seq_length = x.size(1)
        self.pe = self.pe.to(x.device)
        return x + self.pe[:seq_length, :]
