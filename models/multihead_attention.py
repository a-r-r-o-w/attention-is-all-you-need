from typing import Optional

import torch
import torch.nn as nn

from .scaled_dot_product_attention import ScaledDotProductAttention

T = torch.Tensor


class MultiHeadAttention(nn.Module):
    r"""Multi-Head Attention (Section 3.2.2 of paper).

    Args:
    """

    def __init__(
        self,
        embedding_size: int,  # `d_model` in paper
        query_key_size: int,  # `d_k` in paper
        value_size: int,  # `d_v` in paper
        num_heads: int,  # `h` in paper
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_final_linear_mha_bias: bool = False,
        temperature: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.embedding_size = embedding_size
        self.query_key_size = query_key_size
        self.value_size = value_size
        self.num_heads = num_heads
        self.query_key_head_dim = query_key_size // num_heads
        self.value_head_dim = value_size // num_heads

        if self.query_key_head_dim * num_heads != query_key_size:
            raise ValueError(
                f"`{self.query_key_head_dim=}` must be divisible by `{num_heads=}`"
            )
        if self.value_head_dim * num_heads != value_size:
            raise ValueError(
                f"`{self.value_head_dim=}` must be divisible by `{num_heads=}`"
            )

        self.linear_query = nn.Linear(
            self.embedding_size, self.query_key_size, bias=use_query_bias
        )
        self.linear_key = nn.Linear(
            self.embedding_size, self.query_key_size, bias=use_key_bias
        )
        self.linear_value = nn.Linear(
            self.embedding_size, self.value_size, bias=use_value_bias
        )

        self.scaled_dot_product_attn = ScaledDotProductAttention(
            query_key_size=query_key_size,
            temperature=temperature,
        )

        self.linear_final = nn.Linear(
            self.value_size, self.embedding_size, bias=use_final_linear_mha_bias
        )

    def _split(self, x: T, split_size: int) -> T:
        # result shape: [batch_size, seq_length, num_heads, split_size]
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, split_size)
        return x

    def _concat(self, x: T, concat_size: int) -> T:
        # result shape: [batch_size, seq_length, num_heads * concat_size]
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads * concat_size)
        return x

    def forward(self, query: T, key: T, value: T, mask: Optional[T] = None) -> T:
        # 1. Linear
        query = self.linear_query(query)
        key = self.linear_key(key)
        value = self.linear_value(value)

        # 2. Scaled Dot Product Attention
        query = self._split(query, self.query_key_head_dim).transpose(1, 2)
        key = self._split(key, self.query_key_head_dim).transpose(1, 2)
        value = self._split(value, self.value_head_dim).transpose(1, 2)

        # x.shape after next line is [batch_size, num_heads, embedding_size, value_size]
        x: T = self.scaled_dot_product_attn(query, key, value, mask)

        # 3. Concat
        x = x.transpose(1, 2).contiguous()
        x = self._concat(x, self.value_head_dim)

        # 4. Linear
        x = self.linear_final(x)

        return x
