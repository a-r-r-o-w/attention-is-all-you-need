from typing import Optional

import torch
import torch.nn as nn

from .multihead_attention import MultiHeadAttention
from .position_wise_ffn import PositionwiseFeedForward

T = torch.Tensor


class EncoderBlock(nn.Module):
    r"""A single encoder block as shown in Figure 1 of the paper.

    Args:
    """

    def __init__(
        self,
        embedding_size: int,  # `d_model` in paper
        query_key_size: int,  # `d_k` in paper
        value_size: int,  # `d_v` in paper
        num_heads: int,  # `h` in paper
        ffn_hidden_dim: int,
        ffn_activation: str = "relu",
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_final_linear_mha_bias: bool = False,
        use_ffn_bias_1: bool = True,
        use_ffn_bias_2: bool = True,
        temperature: Optional[float] = None,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.mha = MultiHeadAttention(
            embedding_size=embedding_size,
            query_key_size=query_key_size,
            value_size=value_size,
            num_heads=num_heads,
            use_query_bias=use_query_bias,
            use_key_bias=use_key_bias,
            use_value_bias=use_value_bias,
            temperature=temperature,
            use_final_linear_mha_bias=use_final_linear_mha_bias,
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(embedding_size)

        self.pffn = PositionwiseFeedForward(
            in_out_dim=embedding_size,
            hidden_dim=ffn_hidden_dim,
            activation=ffn_activation,
            use_bias_1=use_ffn_bias_1,
            use_bias_2=use_ffn_bias_2,
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embedding_size)

    def forward(self, x: T, mask: Optional[T] = None) -> T:
        # 1. MultiHead Attention
        residual = x.clone()
        x = self.mha(x, x, x, mask)

        # 2. Dropout, Residual addition and Normalization 1
        x = self.dropout1(x)
        x = x + residual
        x = self.norm1(x)

        # 3. Positionwise FFN
        residual = x.clone()
        x = self.pffn(x)

        # 4. Dropout, Residual addition and Normalization 2
        x = self.dropout2(x)
        x = x + residual
        x = self.norm2(x)

        return x


class DecoderBlock(nn.Module):
    r"""A single decoder block as shown in Figure 1 of the paper.

    Args:
    """

    def __init__(
        self,
        embedding_size: int,  # `d_model` in paper
        query_key_size: int,  # `d_k` in paper
        value_size: int,  # `d_v` in paper
        num_heads: int,  # `h` in paper
        ffn_hidden_dim: int,
        ffn_activation: str = "relu",
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_final_linear_mha_bias: bool = False,
        use_ffn_bias_1: bool = True,
        use_ffn_bias_2: bool = True,
        temperature: Optional[float] = None,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.mha1 = MultiHeadAttention(
            embedding_size=embedding_size,
            query_key_size=query_key_size,
            value_size=value_size,
            num_heads=num_heads,
            use_query_bias=use_query_bias,
            use_key_bias=use_key_bias,
            use_value_bias=use_value_bias,
            temperature=temperature,
            use_final_linear_mha_bias=use_final_linear_mha_bias,
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(embedding_size)

        self.mha2 = MultiHeadAttention(
            embedding_size=embedding_size,
            query_key_size=query_key_size,
            value_size=value_size,
            num_heads=num_heads,
            use_query_bias=use_query_bias,
            use_key_bias=use_key_bias,
            use_value_bias=use_value_bias,
            temperature=temperature,
            use_final_linear_mha_bias=use_final_linear_mha_bias,
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embedding_size)

        self.pffn = PositionwiseFeedForward(
            in_out_dim=embedding_size,
            hidden_dim=ffn_hidden_dim,
            activation=ffn_activation,
            use_bias_1=use_ffn_bias_1,
            use_bias_2=use_ffn_bias_2,
        )
        self.dropout3 = nn.Dropout(dropout_rate)
        self.norm3 = nn.LayerNorm(embedding_size)

    def forward(
        self, x: T, enc_x: T, mask: Optional[T] = None, dec_enc_mask: Optional[T] = None
    ) -> T:
        # 1. Masked MultiHead Attention
        residual = x.clone()
        x = self.mha1(x, x, x, mask)

        # 2. Dropout, Residual addition and Normalization 1
        x = self.dropout1(x)
        x = x + residual
        x = self.norm1(x)

        # 3. MultiHead Encoder-Decoder Attention
        residual = x.clone()
        x = self.mha2(x, enc_x, enc_x, dec_enc_mask)

        # 4. Dropout, Residual addition and Normalization 2
        x = self.dropout2(x)
        x = x + residual
        x = self.norm2(x)

        # 5. Positionwise FFN
        residual = x.clone()
        x = self.pffn(x)

        # 6. Dropout, Residual addition and Normalization 3
        x = self.dropout3(x)
        x = x + residual
        x = self.norm3(x)

        return x
