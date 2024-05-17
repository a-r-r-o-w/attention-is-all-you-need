from typing import Optional

import torch
import torch.nn as nn

from .scaled_dot_product_attention import ScaledDotProductAttention

T = torch.FloatTensor


class MultiHeadAttention(nn.Module):
    r"""Multi-Head Attention (Section 3.2.2 of paper).

    Args:
        embedding_dim (`int`):
            The dimension of the embedding space (d_model in paper).
        query_key_dim (`int`):
            The dimension of the query and key vectors (d_k in paper).
        value_dim (`int`):
            The dimension of the value vectors (d_v in paper).
        num_heads (`int`):
            The number of heads (h in paper).
        use_query_bias (`bool`, optional):
            Whether to use bias in the query linear layer. Defaults to `False`. In theory,
            this should have no effect because normalization should remove the effect of bias.
        use_key_bias (`bool`, optional):
            Whether to use bias in the key linear layer. Defaults to `False`. In theory,
            this should have no effect because normalization should remove the effect of bias.
        use_value_bias (`bool`, optional):
            Whether to use bias in the value linear layer. Defaults to `False`. In theory,
            this should have no effect because normalization should remove the effect of bias.
        use_final_linear_mha_bias (`bool`, optional):
            Whether to use bias in the final linear layer of multi-head attention. Defaults to `False`.
    """

    def __init__(
        self,
        embedding_dim: int,  # `d_model` in paper
        query_key_dim: int,  # `d_k` in paper
        value_dim: int,  # `d_v` in paper
        num_heads: int,  # `h` in paper
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_final_linear_mha_bias: bool = False,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.query_key_dim = query_key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.query_key_dim_per_head = query_key_dim // num_heads
        self.value_dim_per_head = value_dim // num_heads

        if self.query_key_dim_per_head * num_heads != query_key_dim:
            raise ValueError(
                f"`{self.query_key_dim_per_head=}` must be divisible by `{num_heads=}`"
            )
        if self.value_dim_per_head * num_heads != value_dim:
            raise ValueError(
                f"`{self.value_dim_per_head=}` must be divisible by `{num_heads=}`"
            )

        self.linear_query = nn.Linear(
            self.embedding_dim, self.query_key_dim, bias=use_query_bias
        )
        self.linear_key = nn.Linear(
            self.embedding_dim, self.query_key_dim, bias=use_key_bias
        )
        self.linear_value = nn.Linear(
            self.embedding_dim, self.value_dim, bias=use_value_bias
        )

        self.scaled_dot_product_attn = ScaledDotProductAttention(
            self.query_key_dim_per_head
        )

        self.linear_final = nn.Linear(
            self.value_dim, self.embedding_dim, bias=use_final_linear_mha_bias
        )

    def forward(self, query: T, key: T, value: T, mask: Optional[T] = None) -> T:
        # 1. Linear
        query = self.linear_query(query)
        key = self.linear_key(key)
        value = self.linear_value(value)

        # 2. Scaled Dot Product Attention
        batch_size, seq_length, _ = query.shape
        query = query.view(batch_size, -1, self.num_heads, self.query_key_dim_per_head)
        key = key.view(batch_size, -1, self.num_heads, self.query_key_dim_per_head)
        value = value.view(batch_size, -1, self.num_heads, self.value_dim_per_head)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # x.shape after next line is [batch_size, num_heads, embedding_dim, value_dim]
        x: T = self.scaled_dot_product_attn(query, key, value, mask)

        # 3. Concat
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.value_dim)

        # 4. Linear
        x = self.linear_final(x)

        return x
