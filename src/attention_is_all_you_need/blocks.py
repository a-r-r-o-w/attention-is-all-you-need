from typing import Optional

import torch
import torch.nn as nn

from .multihead_attention import MultiHeadAttention
from .position_wise_ffn import PositionwiseFeedForward

T = torch.FloatTensor


class EncoderBlock(nn.Module):
    r"""A single encoder block as shown in Figure 1 of the paper.

    Args:
        embedding_dim (`int`):
            The dimension of the embedding space (d_model in paper).
        query_key_dim (`int`):
            The dimension of the query and key vectors (d_k in paper).
        value_dim (`int`):
            The dimension of the value vectors (d_v in paper).
        num_heads (`int`):
            The number of heads (h in paper).
        ffn_hidden_dim (`int`):
            The dimension of the hidden layer in the position-wise feed-forward network.
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
        use_pffn_bias (`bool`, optional):
            Whether to use bias in the position-wise feed-forward network. Defaults to `True`.
        dropout_rate (`float`, optional):
            The dropout rate. Defaults to `0.1`.
    """

    def __init__(
        self,
        embedding_dim: int,  # `d_model` in paper
        query_key_dim: int,  # `d_k` in paper
        value_dim: int,  # `d_v` in paper
        num_heads: int,  # `h` in paper
        ffn_hidden_dim: int,
        ffn_activation: str = "relu",
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_final_linear_mha_bias: bool = False,
        use_pffn_bias: bool = True,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.mha = MultiHeadAttention(
            embedding_dim=embedding_dim,
            query_key_dim=query_key_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            use_query_bias=use_query_bias,
            use_key_bias=use_key_bias,
            use_value_bias=use_value_bias,
            use_final_linear_mha_bias=use_final_linear_mha_bias,
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.pffn = PositionwiseFeedForward(
            in_out_dim=embedding_dim,
            hidden_dim=ffn_hidden_dim,
            activation=ffn_activation,
            use_bias=use_pffn_bias,
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: T, mask: Optional[T] = None) -> T:
        # 1. MultiHead Attention
        residual = x
        x = self.mha(x, x, x, mask)

        # 2. Dropout, Residual addition and Normalization 1
        x = self.dropout1(x)
        x = x + residual
        x = self.norm1(x)

        # 3. Positionwise FFN
        residual = x
        x = self.pffn(x)

        # 4. Dropout, Residual addition and Normalization 2
        x = self.dropout2(x)
        x = x + residual
        x = self.norm2(x)

        return x


class DecoderBlock(nn.Module):
    r"""A single decoder block as shown in Figure 1 of the paper.

    Args:
        embedding_dim (`int`):
            The dimension of the embedding space (d_model in paper).
        query_key_dim (`int`):
            The dimension of the query and key vectors (d_k in paper).
        value_dim (`int`):
            The dimension of the value vectors (d_v in paper).
        num_heads (`int`):
            The number of heads (h in paper).
        ffn_hidden_dim (`int`):
            The dimension of the hidden layer in the position-wise feed-forward network.
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
        use_pffn_bias (`bool`, optional):
            Whether to use bias in the position-wise feed-forward network. Defaults to `True`.
        dropout_rate (`float`, optional):
            The dropout rate. Defaults to `0.1`.
    """

    def __init__(
        self,
        embedding_dim: int,  # `d_model` in paper
        query_key_dim: int,  # `d_k` in paper
        value_dim: int,  # `d_v` in paper
        num_heads: int,  # `h` in paper
        ffn_hidden_dim: int,
        ffn_activation: str = "relu",
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_final_linear_mha_bias: bool = False,
        use_pffn_bias: bool = True,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.mha1 = MultiHeadAttention(
            embedding_dim=embedding_dim,
            query_key_dim=query_key_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            use_query_bias=use_query_bias,
            use_key_bias=use_key_bias,
            use_value_bias=use_value_bias,
            use_final_linear_mha_bias=use_final_linear_mha_bias,
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.mha2 = MultiHeadAttention(
            embedding_dim=embedding_dim,
            query_key_dim=query_key_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            use_query_bias=use_query_bias,
            use_key_bias=use_key_bias,
            use_value_bias=use_value_bias,
            use_final_linear_mha_bias=use_final_linear_mha_bias,
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.pffn = PositionwiseFeedForward(
            in_out_dim=embedding_dim,
            hidden_dim=ffn_hidden_dim,
            activation=ffn_activation,
            use_bias=use_pffn_bias,
        )
        self.dropout3 = nn.Dropout(dropout_rate)
        self.norm3 = nn.LayerNorm(embedding_dim)

    def forward(
        self, x: T, enc_x: T, mask: Optional[T] = None, dec_enc_mask: Optional[T] = None
    ) -> T:
        # 1. Masked MultiHead Attention
        residual = x
        x = self.mha1(x, x, x, mask)

        # 2. Dropout, Residual addition and Normalization 1
        x = self.dropout1(x)
        x = x + residual
        x = self.norm1(x)

        # 3. MultiHead Encoder-Decoder Attention
        residual = x
        x = self.mha2(x, enc_x, enc_x, dec_enc_mask)

        # 4. Dropout, Residual addition and Normalization 2
        x = self.dropout2(x)
        x = x + residual
        x = self.norm2(x)

        # 5. Positionwise FFN
        residual = x
        x = self.pffn(x)

        # 6. Dropout, Residual addition and Normalization 3
        x = self.dropout3(x)
        x = x + residual
        x = self.norm3(x)

        return x
