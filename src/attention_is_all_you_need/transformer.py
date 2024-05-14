from typing import Optional

import torch
import torch.nn as nn

from .blocks import DecoderBlock, EncoderBlock
from .positional_encoding import PositionalEncoding

T = torch.FloatTensor


class EncoderDecoderTransformer(nn.Module):
    r"""Transformer - the model proposed in "Attention Is All You Need".
    Paper: https://arxiv.org/abs/1706.03762
    Original Code: https://github.com/tensorflow/tensor2tensor

    This implementation is an almost close replica of the original transformer model. I've made assumptions
    about some details that are not clear from reading the paper and so the overall number of parameters may
    or may not match completely with the model developed in the original codebase.

    Args:
        num_encoder_layers (`int`):
            The number of encoder layers.
        num_decoder_layers (`int`):
            The number of decoder layers.
        vocab_src_size (`int`):
            The size of the source vocabulary.
        vocab_tgt_size (`int`):
            The size of the target vocabulary.
        pad_src_idx (`int`):
            The index of the padding token in the source vocabulary.
        pad_tgt_idx (`int`):
            The index of the padding token in the target vocabulary.
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
        max_length (`int`, optional):
            The maximum length of the sequence. Defaults to `10000`.
    """

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        vocab_src_size: int,
        vocab_tgt_size: int,
        pad_src_idx: int,
        pad_tgt_idx: int,
        embedding_dim: int,  # `d_model` in paper
        query_key_dim: int,  # `d_k` in paper
        value_dim: int,  # `d_v` in paper
        num_heads: int,  # `h` in paper
        ffn_hidden_dim: int,
        ffn_activation: str = "relu",
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_pffn_bias: bool = True,
        use_final_linear_bias: bool = False,
        dropout_rate: float = 0.1,
        max_length: int = 10000,
    ) -> None:
        super().__init__()

        self.pad_src_idx = pad_src_idx
        self.pad_tgt_idx = pad_tgt_idx

        self.pe = PositionalEncoding(
            embedding_dim=embedding_dim,
            max_length=max_length,
        )

        self.src_emb = nn.Embedding(vocab_src_size, embedding_dim)
        self.tgt_emb = nn.Embedding(vocab_tgt_size, embedding_dim)
        self.scale = torch.sqrt(torch.tensor(embedding_dim, dtype=torch.float32))

        self.src_dropout = nn.Dropout(dropout_rate)
        self.tgt_dropout = nn.Dropout(dropout_rate)

        self.encoder = nn.ModuleList(
            [
                EncoderBlock(
                    embedding_dim=embedding_dim,
                    query_key_dim=query_key_dim,
                    value_dim=value_dim,
                    num_heads=num_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    ffn_activation=ffn_activation,
                    use_query_bias=use_query_bias,
                    use_key_bias=use_key_bias,
                    use_value_bias=use_value_bias,
                    use_final_linear_mha_bias=use_final_linear_bias,
                    use_pffn_bias=use_pffn_bias,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                DecoderBlock(
                    embedding_dim=embedding_dim,
                    query_key_dim=query_key_dim,
                    value_dim=value_dim,
                    num_heads=num_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    ffn_activation=ffn_activation,
                    use_query_bias=use_query_bias,
                    use_key_bias=use_key_bias,
                    use_value_bias=use_value_bias,
                    use_final_linear_mha_bias=use_final_linear_bias,
                    use_pffn_bias=use_pffn_bias,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.linear = nn.Linear(embedding_dim, vocab_tgt_size)

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
        seq_length = x.size(1)
        pad_mask = (x != pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.ones((1, seq_length, seq_length), device=x.device)
        causal_mask = torch.tril(causal_mask).bool()
        mask = pad_mask & causal_mask
        return mask

    def forward(self, src_x: T, tgt_x: T) -> T:
        memory = self.encode(src_x)
        x = self.decode(src_x, tgt_x, memory)
        return x

    def encode(self, src_x: T) -> T:
        r"""Forward pass through encoder.

        Args:
            src_x (`torch.Tensor`): The source tokens ids.

        Returns:
            `torch.Tensor`: The output of the encoder.
        """
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
        for block in self.encoder:
            src_x = block(src_x, src_mask)

        return src_x

    def decode(self, src_x: T, tgt_x: T, enc_x: T) -> T:
        r"""Forward pass through decoder.

        Args:
            src_x (`torch.Tensor`): The source tokens ids.
            tgt_x (`torch.Tensor`): The target tokens ids.
            enc_x (`torch.Tensor`): The output of the encoder.

        Returns:
            `torch.Tensor`: The output of the decoder.
        """
        # 1. Prepare masks for decoder
        src_mask = self._get_src_mask(src_x, self.pad_src_idx)
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
        for block in self.decoder:
            tgt_x = block(tgt_x, enc_x, tgt_mask, src_mask)

        # 6. Linear projection to get probabilities for output tokens using softmax
        x = self.linear(tgt_x)

        return x
