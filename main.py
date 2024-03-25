import os
from typing import Optional

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from data import Multi30kDatasetHandler, Vocabulary
from models import Transformer, PositionalEncoding, LRScheduler
from utils import get_summary, initialize_weights, collate_fn


def _print_with_line(content: str, line_length: int = 80):
    print(content)
    print("-" * line_length)


def seq_to_seq_translate(
    transformer: nn.Module,
    en_tensors: torch.Tensor,
    de_tensors: torch.Tensor,
    vocab: Vocabulary,
    sos_token: str,
    max_length: int,
):
    en_tensors = en_tensors.to(device="cuda")
    de_tensors = de_tensors.to(device="cuda")
    current_batch_size = en_tensors.shape[0]
    outputs = torch.zeros(current_batch_size, max_length).type_as(en_tensors)
    outputs[:, 0] = torch.LongTensor(
        [vocab.stoi[sos_token]] * current_batch_size
    )  # start with <sos> token

    for t in range(1, max_length):
        output = transformer(en_tensors, outputs[:, :t])
        output = output.argmax(dim=-1)
        outputs[:, t] = output[:, -1]

    sep = " "
    targets = []
    generated = []
    for i in range(current_batch_size):
        targets.append(sep.join([vocab.itos[idx.item()] for idx in de_tensors[i, :]]))
        generated.append(sep.join([vocab.itos[idx.item()] for idx in outputs[i, :]]))

    return targets, generated


class CLI:
    r"""Command-line interface to interact with the transformer implementation for
    training or inference.
    """

    def __init__(self) -> None:
        pass

    def train(
        self,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        vocab_src_size: int = 25000,
        vocab_tgt_size: int = 25000,
        pad_src_idx: int = 24999,
        pad_tgt_idx: int = 24999,
        embedding_size: int = 512,  # `d_model` in paper
        query_key_size: int = 64,  # `d_k` in paper
        value_size: int = 64,  # `d_v` in paper
        num_heads: int = 8,  # `h` in paper
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
        batch_size: int = 32,
        dataset_name: str = "multi30k",
        tokenizer_type: str = "spacy",
        tokenizer_vocab_size: int = 3000,
        tokenizer_min_frequency: int = 2,
        epochs: int = 10,
        seed: int = 42,
        validation_epochs: int = 1,
        checkpoint_path: str = "checkpoints",
        checkpoint_name: str = "transformer",
        checkpoint_steps: int = 500,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        r"""Train the transformer model. You can configure various hyperparameters.

        Args:
            num_layers:
                Number of encoder/decoder layers to be used in the transformer.
        """

        torch.manual_seed(seed)
        np.random.seed(seed)

        checkpoint_path = checkpoint_path

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)

        match dataset_name:
            case "multi30k":
                dataset = Multi30kDatasetHandler(
                    path="dataset/multi30k",
                    files={
                        "train": "train.jsonl",
                        "test": "test.jsonl",
                        "val": "val.jsonl",
                    },
                    sos_token="<sos>",
                    eos_token="<eos>",
                    unk_token="<unk>",
                    pad_token="<pad>",
                    max_length=max_length,
                    tokenizer_type=tokenizer_type,
                    tokenizer_kwargs=None
                    if tokenizer_type != "bpe"
                    else {
                        "vocab_size": tokenizer_vocab_size,
                        "min_frequency": tokenizer_min_frequency,
                    },
                    ignorecase=True,
                )
            case _:
                raise ValueError(f"Dataset {dataset_name} not supported")

        def collate_helper(batch):
            return collate_fn(
                batch,
                dataset.en_vocab.stoi[dataset.pad_token],
                dataset.de_vocab.stoi[dataset.pad_token],
            )

        train_dataset, test_dataset, val_dataset = dataset.get_datasets()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_helper,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_helper,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_helper,
        )

        en_vocab_size = len(dataset.en_vocab)
        de_vocab_size = len(dataset.de_vocab)

        print(f"{en_vocab_size=}")
        print(f"{de_vocab_size=}")

        if vocab_src_size == -1:
            vocab_src_size = len(train_dataset.en_vocab)
        if vocab_tgt_size == -1:
            vocab_tgt_size = len(train_dataset.de_vocab)
        if pad_src_idx == -1:
            pad_src_idx = train_dataset.en_vocab.stoi[train_dataset.pad_token]
        if pad_tgt_idx == -1:
            pad_tgt_idx = train_dataset.de_vocab.stoi[train_dataset.pad_token]

        transformer = Transformer(
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
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
        ).to(device="cuda")
        transformer.train()

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
        # lr_scheduler = LRScheduler(
        #     optimizer, embedding_size=embedding_size, warmup_steps=4000
        # )

        criterion = nn.CrossEntropyLoss(ignore_index=pad_tgt_idx)

        train_losses = []
        val_losses = []
        learning_rates = []
        step = 0
        total_steps = len(train_dataloader) * epochs

        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        (line1,) = ax1.plot([], [], "r-")  # line for train loss
        (line2,) = ax1.plot([], [], "g-")  # line for validation loss

        ax2.set_xlabel("Step")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate")
        (line3,) = ax2.plot([], [], "b-")  # line for learning rate

        with tqdm(total=total_steps, desc="Training") as trainbar:
            for epoch in range(1, epochs + 1):
                total_loss = 0.0

                transformer.train()

                for i, (en_tensors, de_tensors) in enumerate(train_dataloader):
                    en_tensors = en_tensors.to(device="cuda")
                    de_tensors = de_tensors.to(device="cuda")
                    optimizer.zero_grad()
                    output = transformer(
                        en_tensors, de_tensors[:, :-1]
                    )  # drop eos token
                    # output = output.permute(
                    #     0, 2, 1
                    # )  # [batch_size, de_vocab_size, max_length]
                    # loss = criterion(output, de_tensors[:, 1:])  # drop sos token
                    loss = criterion(
                        output.contiguous().view(-1, de_vocab_size),
                        de_tensors[:, 1:].contiguous().view(-1),
                    )  # drop sos token

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1)

                    if (
                        step + 1 == total_steps
                        or step % gradient_accumulation_steps == 0
                    ):
                        for param in transformer.parameters():
                            if param.grad is not None:
                                param.grad /= gradient_accumulation_steps
                        optimizer.step()
                        optimizer.zero_grad()

                    total_loss += loss.item()
                    # learning_rates.append(lr_scheduler.get_lr())
                    # lr_scheduler.step()

                    step += 1
                    trainbar.update()

                    if step % checkpoint_steps == 0:
                        torch.save(
                            transformer.state_dict(),
                            os.path.join(
                                checkpoint_path, f"{checkpoint_name}_{step}.pth"
                            ),
                        )

                train_losses.append(total_loss / len(train_dataloader))
                print()
                print(f"Epoch: {epoch}")
                print(
                    f"Train Loss: [{total_loss=:.3f}] {total_loss / len(train_dataloader):.3f}"
                )
                print(f"Perplexity: {np.exp(total_loss / len(train_dataloader)):.3f}")
                print()

                # val set
                if (epoch - 1) % validation_epochs == 0:
                    total_loss = 0.0
                    transformer.eval()
                    with torch.no_grad():
                        with tqdm(
                            total=len(val_dataloader), desc="Validation"
                        ) as valbar:
                            for i, (en_tensors, de_tensors) in enumerate(
                                val_dataloader
                            ):
                                en_tensors = en_tensors.to(device="cuda")
                                de_tensors = de_tensors.to(device="cuda")
                                output = transformer(
                                    en_tensors, de_tensors[:, :-1]
                                )  # drop eos token
                                # output = output.permute(
                                #     0, 2, 1
                                # )  # [batch_size, de_vocab_size, max_length]
                                # loss = criterion(output, de_tensors[:, 1:])  # drop sos token
                                loss = criterion(
                                    output.contiguous().view(-1, de_vocab_size),
                                    de_tensors[:, 1:].contiguous().view(-1),
                                )  # drop sos token
                                total_loss += loss.item()
                                valbar.update()

                    val_losses.append(total_loss / len(val_dataloader))
                    print()
                    print(
                        f"Validation Loss: [{total_loss=:.3f}] {total_loss / len(val_dataloader):.3f}"
                    )
                    print(f"Perplexity: {np.exp(total_loss / len(val_dataloader)):.3f}")
                    print()

                transformer.eval()
                with torch.no_grad():
                    for i, (en_tensors, de_tensors) in enumerate(test_dataloader):
                        en_tensors = en_tensors[:1]
                        de_tensors = de_tensors[:1]
                        targets, generated = seq_to_seq_translate(
                            transformer,
                            en_tensors,
                            de_tensors,
                            dataset.de_vocab,
                            dataset.sos_token,
                            max_length,
                        )
                        print("Running testset inference")
                        print(f"   target: {targets[0]}")
                        print(f"generated: {generated[0]}")
                        print()
                        break

                line1.set_xdata(range(1, len(train_losses) + 1))
                line1.set_ydata(train_losses)
                line2.set_xdata(range(1, len(val_losses) + 1))
                line2.set_ydata(val_losses)
                ax1.relim()
                ax1.autoscale_view()

                line3.set_xdata(range(1, len(learning_rates) + 1))
                line3.set_ydata(learning_rates)
                ax2.relim()
                ax2.autoscale_view()

                # Redraw the figure
                fig.canvas.draw()
                fig.canvas.flush_events()

        plt.ioff()
        plt.show()

        torch.save(
            transformer.state_dict(),
            os.path.join(checkpoint_path, f"{checkpoint_name}_final.pth"),
        )
        np.save("train_losses.npy", np.array(train_losses))
        np.save("val_losses.npy", np.array(val_losses))
        np.save("learning_rates.npy", np.array(learning_rates))

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
