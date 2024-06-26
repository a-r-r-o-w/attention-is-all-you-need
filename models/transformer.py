from typing import Optional

import torch
import torch.nn as nn

from .blocks import DecoderBlock, EncoderBlock
from .positional_encoding import PositionalEncoding

T = torch.FloatTensor


class Encoder(nn.Module):
    r"""Stack of encoder blocks used in the transformer (see Figure 1 in paper).

    Args:
    """

    def __init__(
        self,
        num_layers: int,
        embedding_size: int,  # `d_model` in paper
        query_key_size: int,  # `d_k` in paper
        value_size: int,  # `d_v` in paper
        num_heads: int,  # `h` in paper
        ffn_hidden_dim: int,
        ffn_activation: str = "relu",
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_ffn_bias_1: bool = True,
        use_ffn_bias_2: bool = True,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    embedding_size=embedding_size,
                    query_key_size=query_key_size,
                    value_size=value_size,
                    num_heads=num_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    ffn_activation=ffn_activation,
                    use_query_bias=use_query_bias,
                    use_key_bias=use_key_bias,
                    use_value_bias=use_value_bias,
                    use_ffn_bias_1=use_ffn_bias_1,
                    use_ffn_bias_2=use_ffn_bias_2,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: T, mask: Optional[T] = None) -> T:
        # 1. Process encoder blocks
        for block in self.blocks:
            x = block(x, mask)

        return x


class Decoder(nn.Module):
    r"""Stack of decoder blocks used in the transformer (see Figure 1 in paper).

    Args:
    """

    def __init__(
        self,
        num_layers: int,
        embedding_size: int,  # `d_model` in paper
        query_key_size: int,  # `d_k` in paper
        value_size: int,  # `d_v` in paper
        num_heads: int,  # `h` in paper
        ffn_hidden_dim: int,
        ffn_activation: str = "relu",
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_ffn_bias_1: bool = True,
        use_ffn_bias_2: bool = True,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    embedding_size=embedding_size,
                    query_key_size=query_key_size,
                    value_size=value_size,
                    num_heads=num_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    ffn_activation=ffn_activation,
                    use_query_bias=use_query_bias,
                    use_key_bias=use_key_bias,
                    use_value_bias=use_value_bias,
                    use_ffn_bias_1=use_ffn_bias_1,
                    use_ffn_bias_2=use_ffn_bias_2,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x: T, enc_x: T, mask: Optional[T] = None, dec_enc_mask: Optional[T] = None
    ) -> T:
        # 1. Process decoder blocks
        for block in self.blocks:
            x = block(x, enc_x, mask, dec_enc_mask)

        return x


class Transformer(nn.Module):
    r"""Transformer - the model proposed in "Attention Is All You Need".
    Paper: https://arxiv.org/abs/1706.03762
    Original Code: https://github.com/tensorflow/tensor2tensor

    This implementation is an almost close replica of the original transformer model. I've made assumptions
    about some details that are not clear from reading the paper and so the overall number of parameters may
    or may not match completely with the model developed in the original codebase.

    Args:
    """

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        vocab_src_size: int,
        vocab_tgt_size: int,
        pad_src_idx: int,
        pad_tgt_idx: int,
        embedding_size: int,  # `d_model` in paper
        query_key_size: int,  # `d_k` in paper
        value_size: int,  # `d_v` in paper
        num_heads: int,  # `h` in paper
        ffn_hidden_dim: int,
        ffn_activation: str = "relu",
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_ffn_bias_1: bool = True,
        use_ffn_bias_2: bool = True,
        use_final_linear_bias: bool = False,
        dropout_rate: float = 0.1,
        max_length: int = 10000,
    ) -> None:
        super().__init__()

        self.pad_src_idx = pad_src_idx
        self.pad_tgt_idx = pad_tgt_idx

        self.pe = PositionalEncoding(
            embedding_size=embedding_size,
            max_length=max_length,
        )

        self.src_emb = nn.Embedding(vocab_src_size, embedding_size)
        self.tgt_emb = nn.Embedding(vocab_tgt_size, embedding_size)
        self.scale = torch.sqrt(torch.tensor(embedding_size, dtype=torch.float32))

        self.src_dropout = nn.Dropout(dropout_rate)
        self.tgt_dropout = nn.Dropout(dropout_rate)

        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            embedding_size=embedding_size,
            query_key_size=query_key_size,
            value_size=value_size,
            num_heads=num_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            ffn_activation=ffn_activation,
            use_query_bias=use_query_bias,
            use_key_bias=use_key_bias,
            use_value_bias=use_value_bias,
            use_ffn_bias_1=use_ffn_bias_1,
            use_ffn_bias_2=use_ffn_bias_2,
            dropout_rate=dropout_rate,
        )

        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            embedding_size=embedding_size,
            query_key_size=query_key_size,
            value_size=value_size,
            num_heads=num_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            ffn_activation=ffn_activation,
            use_query_bias=use_query_bias,
            use_key_bias=use_key_bias,
            use_value_bias=use_value_bias,
            use_ffn_bias_1=use_ffn_bias_1,
            use_ffn_bias_2=use_ffn_bias_2,
            dropout_rate=dropout_rate,
        )

        self.linear = nn.Linear(
            embedding_size, vocab_tgt_size, bias=use_final_linear_bias
        )

    def _get_src_mask(self, x: T, pad_idx: int) -> torch.BoolTensor:
        r"""Helper utility to get mask for padded tokens. Padded tokens should not be paid attention."""
        pad_mask = (x != pad_idx).bool().unsqueeze(1).unsqueeze(2)
        return pad_mask

    def _get_tgt_mask(self, x: T, pad_idx: int) -> torch.BoolTensor:
        r"""Helper utility to get mask for decoder. The decoder should not pay attention to future tokens.

        This returns a tensor that looks like:
            [
                [1, 0, 0, ...],
                [1, 1, 0, ...],
                [1, 1, 1, ...],
                ...
            ]
        """
        # batch_size = x.size(0)
        seq_length = x.size(1)
        pad_mask = (x != pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.ones((1, seq_length, seq_length), device=x.device)
        causal_mask = torch.tril(causal_mask).bool()
        mask = pad_mask & causal_mask
        return mask

    def forward(self, src_x: T, tgt_x: T) -> T:
        # 1. Prepare masks for encoder and decoder
        src_mask = self._get_src_mask(src_x, self.pad_src_idx)
        tgt_mask = self._get_tgt_mask(tgt_x, self.pad_tgt_idx)

        # 2. Convert tokens to embeddings
        src_x = self.src_emb(src_x)
        tgt_x = self.tgt_emb(tgt_x)

        # 3. Apply positional encoding
        self.scale = self.scale.to(src_x.device, dtype=src_x.dtype)
        src_x = src_x * self.scale
        tgt_x = tgt_x * self.scale
        src_x = self.pe(src_x)
        tgt_x = self.pe(tgt_x)

        # 4. Regularization after embed as described in section 5.4 of the paper
        src_x = self.src_dropout(src_x)
        tgt_x = self.tgt_dropout(tgt_x)

        # 5. Forward pass through encoder, and final outputs of encoder
        #    used to condition all decoder layers
        x = self.encoder(src_x, src_mask)
        x = self.decoder(tgt_x, x, tgt_mask, src_mask)

        # 6. Linear projection to get probabilities for output tokens using softmax
        x = self.linear(x)

        return x

    def encode(self, src_x: T) -> T:
        # 1. Prepare masks for encoder
        src_mask = self._get_src_mask(src_x, self.pad_src_idx)

        # 2. Convert tokens to embeddings
        src_x = self.src_emb(src_x)

        # 3. Apply positional encoding
        self.scale = self.scale.to(src_x.device, dtype=src_x.dtype)
        src_x = src_x * self.scale
        src_x = self.pe(src_x)

        # 4. Regularization after embed as described in section 5.4 of the paper
        src_x = self.src_dropout(src_x)

        # 5. Forward pass through encoder
        x = self.encoder(src_x, src_mask)

        return x

    def decode(self, tgt_x: T, enc_x: T) -> T:
        # 1. Prepare masks for decoder
        tgt_mask = self._get_tgt_mask(tgt_x, self.pad_tgt_idx)

        # 2. Convert tokens to embeddings
        tgt_x = self.tgt_emb(tgt_x)

        # 3. Apply positional encoding
        self.scale = self.scale.to(tgt_x.device, dtype=tgt_x.dtype)
        tgt_x = tgt_x * self.scale
        tgt_x = self.pe(tgt_x)

        # 4. Regularization after embed as described in section 5.4 of the paper
        tgt_x = self.tgt_dropout(tgt_x)

        # 5. Forward pass through decoder
        x = self.decoder(tgt_x, enc_x, tgt_mask)

        # 6. Linear projection to get probabilities for output tokens using softmax
        x = self.linear(x)

        return x
