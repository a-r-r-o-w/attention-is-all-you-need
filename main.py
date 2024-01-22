from typing import Optional

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from PIL import Image

from models import Transformer, PositionalEncoding
from utils import get_summary, initialize_weights


def _print_with_line(content: str, line_length: int = 80):
    print(content)
    print("-" * line_length)


class CLI:
    r"""Command-line interface to interact with the transformer implementation for
    training or inference.
    """
    
    def __init__(self) -> None:
        pass

    def train(
        self,
        num_layers: int = 6,
        vocab_src_size: int = 25000,
        vocab_tgt_size: int = 25000,
        pad_src_idx: int = 24999,
        pad_tgt_idx: int = 24999,
        embedding_size: int = 512, # `d_model` in paper
        query_key_size: int = 64,  # `d_k` in paper
        value_size: int = 64,      # `d_v` in paper
        num_heads: int = 8,        # `h` in paper
        ffn_hidden_dim: int = 2048,
        ffn_activation: str = "relu",
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_ffn_bias_1: bool = True,
        use_ffn_bias_2: bool = True,
        use_final_linear_bias: bool = False,
        temperature: Optional[float] = None,
        dropout_rate: float = 0.1,
        max_length: int = 10000,
        weight_initialization_method: str = "kaiming_uniform",
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-4,
    ) -> None:
        r"""Train the transformer model. You can configure various hyperparameters.

        Args:
            num_layers:
                Number of encoder/decoder layers to be used in the transformer.
        """

        transformer = Transformer(
            num_layers=num_layers,
            vocab_src_size=vocab_src_size,
            vocab_tgt_size=vocab_tgt_size,
            pad_src_idx=pad_src_idx,
            pad_tgt_idx=pad_tgt_idx,
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
            use_final_linear_bias=use_final_linear_bias,
            temperature=temperature,
            dropout_rate=dropout_rate,
            max_length=max_length,
        )

        initialize_weights(transformer, weight_initialization_method)

        _print_with_line(transformer)
        _print_with_line(f"Summary:\n{get_summary(transformer)}")

        optimizer = optim.Adam(
            params=transformer.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            # based on section 5.3 of paper
            betas=(0.9, 0.98), 
            eps=1e-9,
        )
    
    def visualize_positional_encoding(
        self,
        embedding_size: int = 64,
        max_length: int = 64,
        *,
        save: bool = False,
        output_path: str = "pe.png",
    ) -> None:
        r"""Visualize positional encoding used in the paper.

        Args:
            embedding_size:
                The dimensionality of vector space embeddings (`d_model` in the paper)
            max_length:
                Maximum sequence length of tokens
            save:
                Whether or not to save the plot
            output_path:
                Path to file where plot is to be saved
        """
        
        position_encoder = PositionalEncoding(embedding_size, max_length)
        pe: np.ndarray = position_encoder.pe.detach().numpy()
        
        figsize = (
            min(embedding_size // 8, 20),
            min(max_length // 8, 20),
        )
        plt.figure(figsize=figsize)
        plt.imshow(pe, cmap="magma")
        plt.xlabel("Embedding size (d_model)", fontsize=20)
        plt.ylabel("Sequence length", fontsize=20)
        plt.title("Positional Encoding", fontsize=20)

        if save:
            plt.savefig(output_path)
        
        plt.show()


if __name__ == "__main__":
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(CLI())
