from typing import Optional

import torch
import torch.nn as nn


T = torch.FloatTensor


class ScaledDotProductAttention(nn.Module):
    r"""ScaledDotProductAttention (Section 3.2.1 of paper).

    Args:
        embedding_dim (`int`):
            The dimension of the embedding space (d_model in paper).
    """

    def __init__(self, query_key_dim: int) -> None:
        super().__init__()

        # In the original paper, product of query and key_T are normalized by square root of
        # embedding size. Here, we allow for normalizing with a temperature value too. If
        # temperature is not `None`, it will be used. Otherwise, square root of `embedding_dim`
        # will be used.
        self.query_key_dim = query_key_dim

        scale = torch.sqrt(torch.FloatTensor([query_key_dim]))
        self.register_buffer("scale", scale)
        self.scale: T

        self.softmax = nn.Softmax(dim=3)

    def forward(self, query: T, key: T, value: T, mask: Optional[T] = None) -> T:
        # 1. MatMul
        #  query: [batch_size, num_heads, seq_length, query_key_dim]
        #  key_T: [batch_size, num_heads, query_key_dim, seq_length]
        # result: [batch_size, num_heads, seq_length, seq_length]
        key_T = key.transpose(2, 3)
        x = torch.matmul(query, key_T)

        # 2. Scale
        x = x / self.scale

        # 3. Mask
        if mask is not None:
            x = x.masked_fill(mask == False, value=-1e9)  # noqa

        # 4. SoftMax
        x = self.softmax(x)

        # 5. MatMul
        #      x: [batch_size, num_heads, seq_length, seq_length]
        #  value: [batch_size, num_heads, seq_length, value_dim]
        # result: [batch_size, num_heads, seq_length, value_dim]
        x = torch.matmul(x, value)

        return x
