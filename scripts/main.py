import json
import os
import random
from typing import List, Tuple, Union

import dotenv
import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from attention_is_all_you_need import (
    EncoderDecoderTransformer,
    PositionalEncoding,
)
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from utils import bleu_score, collate_fn, get_summary, initialize_weights


T = torch.Tensor

dotenv.load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")


def _print_with_line(content: str, line_length: int = 80):
    print(content)
    print("-" * line_length)


def seed_everything(seed: int, cuda_deterministic: bool = False):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
        src_vocab_size: int = 25000,
        tgt_vocab_size: int = 25000,
        embedding_dim: int = 512,
        query_key_dim: int = 512,
        value_dim: int = 512,
        num_heads: int = 8,
        ffn_hidden_dim: int = 2048,
        ffn_activation: str = "relu",
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_final_linear_bias: bool = False,
        use_pffn_bias: bool = True,
        dropout_rate: float = 0.1,
        max_length: int = 10000,
        weight_initialization_method: str = "kaiming_uniform",
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        dataset_name: str = "multi30k",
        epochs: int = 10,
        seed: int = 42,
        checkpoint_path: str = "checkpoints",
        experiment_name: str = "transformer",
        checkpoint_steps: int = 500,
        gradient_accumulation_steps: int = 1,
        device: str = "cuda:0",
        track_wandb: bool = False,
    ) -> None:
        r"""Train the transformer model. You can configure various hyperparameters.

        Args:
            num_encoder_layers (int, defaults to *6*):
                Number of encoder layers to be used in the transformer.
            num_decoder_layers (int, defaults to *6*):
                Number of decoder layers to be used in the transformer.
            src_vocab_size (int, defaults to *25000*):
                Vocabulary size for source language after tokenizing with Byte-pair encoding.
            tgt_vocab_size (int, defaults to *25000*):
                Vocabulary size for target language after tokenizing with Byte-pair encoding.
            pad_src_idx (int, defaults to *24999*):
                Index of padding token for source language.
            pad_tgt_idx (int, defaults to *24999*):
                Index of padding token for target language.
            embedding_dim (int, defaults to *512*):
                The dimension of the embedding space (d_model in paper).
            query_key_dim (int, defaults to *64*):
                The dimension of the query and key vectors (d_k in paper).
            value_dim (int, defaults to *64*):
                The dimension of the value vectors (d_v in paper).
            num_heads (int, defaults to *8*):
                The number of heads (h in paper).
            ffn_hidden_dim (int, defaults to *2048*):
                The dimension of the hidden layer in the position-wise feed-forward network.
            ffn_activation (str, defaults to *relu*):
                The activation function to use in the position-wise feed-forward network. Can be one of
                "relu", "gelu", "silu"/"swish", "leaky_relu", "sigmoid", "glu", "reglu", "geglu", or "swiglu".
            use_query_bias (bool, defaults to *False*):
                Whether to use bias in the query linear layer.
            use_key_bias (bool, defaults to *False*):
                Whether to use bias in the key linear layer.
            use_value_bias (bool, defaults to *False*):
                Whether to use bias in the value linear layer.
            use_final_linear_bias (bool, defaults to *False*):
                Whether to use bias in the final linear layer of multi-head attention.
            use_pffn_bias (bool, defaults to *True*):
                Whether to use bias in the position-wise feed-forward network.
            dropout_rate (float, defaults to *0.1*):
                The dropout rate.
            max_length (int, defaults to *10000*):
                The maximum length of any given sequence.
            weight_initialization_method (str, defaults to *kaiming_uniform*):
                The weight initialization method to use. Can be one of "kaiming_uniform", "kaiming_normal",
                "xavier_uniform", "xavier_normal", "uniform", or "normal".
            learning_rate (float, defaults to *1e-5*):
                The learning rate for the optimizer.
            weight_decay (float, defaults to *1e-4*):
                The weight decay for the optimizer.
            batch_size (int, defaults to *32*):
                The batch size for training.
            dataset_name (str, defaults to *"multi30k"*):
                The dataset to use for training. Currently, only "multi30k" is supported.
            epochs (int, defaults to *10*):
                The number of epochs to train the model.
            seed (int, defaults to *42*):
                The random seed to use for reproducibility.
            validation_epochs (int, defaults to *1*):
                The number of epochs after which to run validation.
            checkpoint_path (str, defaults to *"checkpoints"*):
                The path where to save the model checkpoints.
            experiment_name (str, defaults to *"transformer"*):
                The name of the experiment.
            checkpoint_steps (int, defaults to *500*):
                The number of steps after which to save the model checkpoint.
            gradient_accumulation_steps (int, defaults to *1*):
                The number of steps to accumulate gradients before updating the model.
            track_wandb (bool, defaults to *False*):
                Whether to track the experiment with wandb.
        """

        config = {k: v for k, v in locals().items() if k != "self"}

        if track_wandb and not WANDB_API_KEY:
            raise ValueError("`WANDB_API_KEY` is not set in the environment variables")
        if track_wandb:
            wandb.login(key=WANDB_API_KEY, relogin=True)
            run = wandb.init(project="attention_is_all_you_need", name=experiment_name, config=config)
            print("Logged in to wandb")

        def wandb_log(log_dict: dict) -> None:
            if track_wandb:
                wandb.log(log_dict)

        seed_everything(seed)

        sos_token = "<sos>"
        eos_token = "<eos>"
        unk_token = "<unk>"
        pad_token = "<pad>"

        experiment_dir = os.path.join(checkpoint_path, experiment_name)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir, exist_ok=True)
        
        path = f"dataset/{dataset_name}"
        files = {
            "train": "train.jsonl",
            "test": "test.jsonl",
        }
        data = {}
        for split, filename in files.items():
            if split not in ["train", "test"]:
                raise ValueError(f"Split '{split}' is not supported")

            data[split] = []

            with open(os.path.join(path, filename), "r") as f:
                for line in f:
                    item = json.loads(line)
                    item["src"] = item["src"].lower()
                    item["tgt"] = item["tgt"].lower()
                    data[split].append(item)

        sentences_src = [item["src"] for split in data.keys() for item in data[split]]
        sentences_tgt = [item["tgt"] for split in data.keys() for item in data[split]]

        tokenizer_src = Tokenizer(BPE(unk_token=unk_token))
        tokenizer_tgt = Tokenizer(BPE(unk_token=unk_token))
        tokenizer_src.pre_tokenizer = Whitespace()
        tokenizer_tgt.pre_tokenizer = Whitespace()

        trainer_src = BpeTrainer(
            special_tokens=[sos_token, eos_token, unk_token, pad_token],
            vocab_size=src_vocab_size,
            min_frequency=2,
        )
        trainer_tgt = BpeTrainer(
            special_tokens=[sos_token, eos_token, unk_token, pad_token],
            vocab_size=tgt_vocab_size,
            min_frequency=2,
        )

        tokenizer_src.train_from_iterator(sentences_src, trainer_src)
        tokenizer_tgt.train_from_iterator(sentences_tgt, trainer_tgt)

        sos_token_idx = tokenizer_src.token_to_id(sos_token)
        eos_token_idx = tokenizer_src.token_to_id(eos_token)
        for split in data.keys():
            data_tensors = []
            for item in data[split]:
                item["src"] = [sos_token_idx] + tokenizer_src.encode(item["src"]).ids + [eos_token_idx]
                item["tgt"] = [sos_token_idx] + tokenizer_tgt.encode(item["tgt"]).ids + [eos_token_idx]
                item["src"] = torch.tensor(item["src"][:max_length], dtype=torch.long)
                item["tgt"] = torch.tensor(item["tgt"][:max_length], dtype=torch.long)
                data_tensors.append(item)
            data[split] = data_tensors

        pad_src_idx = tokenizer_src.token_to_id(pad_token)
        pad_tgt_idx = tokenizer_tgt.token_to_id(pad_token)

        def collate_helper(batch):
            return collate_fn(
                batch,
                src_pad_token_id=pad_src_idx,
                tgt_pad_token_id=pad_tgt_idx,
                max_length=max_length,
            )

        train_dataloader = DataLoader(
            [(item["src"], item["tgt"]) for item in data["train"]],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_helper,
        )
        test_dataloader = DataLoader(
            [(item["src"], item["tgt"]) for item in data["test"]],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_helper,
        )

        transformer = EncoderDecoderTransformer(
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            pad_src_idx=pad_src_idx,
            pad_tgt_idx=pad_tgt_idx,
            embedding_dim=embedding_dim,
            query_key_dim=query_key_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            ffn_activation=ffn_activation,
            use_query_bias=use_query_bias,
            use_key_bias=use_key_bias,
            use_value_bias=use_value_bias,
            use_pffn_bias=use_pffn_bias,
            use_final_linear_bias=use_final_linear_bias,
            dropout_rate=dropout_rate,
            max_length=max_length,
        ).to(device=device)

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
        #     optimizer, embedding_dim=embedding_dim, warmup_steps=4000
        # )

        criterion = nn.CrossEntropyLoss(ignore_index=pad_tgt_idx)

        if track_wandb:
            wandb.watch(transformer, log="all", log_freq=1000)

        train_losses, test_losses = [], []
        train_bleu_scores, test_bleu_scores = [], []
        step = 0
        total_steps = len(train_dataloader) * epochs

        def perform_forward(src_tensors, tgt_tensors) -> Tuple[T, T]:
            src_tensors = src_tensors.to(device=device)
            tgt_tensors = tgt_tensors.to(device=device)
            src_de = tgt_tensors[:, :-1]
            tgt_de = tgt_tensors[:, 1:].contiguous().view(-1)

            optimizer.zero_grad()
            output = transformer(src_tensors, src_de)
            loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_de)
            return output, loss

        def calculate_bleu_score(output_tensors, tgt_tensors):
            output_tokens = tokenizer_tgt.decode_batch(
                output_tensors.argmax(dim=-1).cpu().numpy(), skip_special_tokens=True
            )
            tgt_tokens = tokenizer_tgt.decode_batch(
                tgt_tensors[:, 1:].cpu().numpy(), skip_special_tokens=True
            )
            tgt_tokens = [[t] for t in tgt_tokens]
            return bleu_score(output_tokens, tgt_tokens)

        with tqdm(total=total_steps, desc="Training") as train_bar:
            for epoch in range(1, epochs + 1):
                total_loss = 0.0
                bleu = 0.0

                transformer.train()
                for i, (src_tensors, tgt_tensors) in enumerate(train_dataloader):
                    output, loss = perform_forward(src_tensors, tgt_tensors)
                    loss.backward()
                    total_loss += loss.item()
                    torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1)
                    bleu += calculate_bleu_score(output, tgt_tensors)

                    if step + 1 == total_steps or step % gradient_accumulation_steps == 0:
                        for param in transformer.parameters():
                            if param.grad is not None:
                                param.grad /= gradient_accumulation_steps
                        optimizer.step()
                        optimizer.zero_grad()

                    step += 1
                    train_bar.update()

                    if step % checkpoint_steps == 0:
                        torch.save(
                            transformer.state_dict(),
                            os.path.join(experiment_dir, f"{experiment_name}_{step}.pth"),
                        )

                train_losses.append(total_loss / len(train_dataloader))
                train_bleu_scores.append(bleu / len(train_dataloader))
                wandb_log(
                    {
                        "train/loss": train_losses[-1],
                        "train/perplexity": np.exp(train_losses[-1]),
                        "train/bleu": train_bleu_scores[-1],
                    }
                )
                print()
                print(f"Epoch: {epoch}")
                print(f"Train Loss: [{total_loss=:.3f}] {train_losses[-1]:.3f}")
                print(f"Perplexity: {np.exp(train_losses[-1]):.3f}")
                print(f"BLEU Score: {train_bleu_scores[-1] * 100:.3f}")
                print()

                total_loss = 0.0
                bleu = 0.0

                transformer.eval()
                with torch.no_grad():
                    with tqdm(total=len(test_dataloader), desc="Testing") as testbar:
                        for i, (src_tensors, tgt_tensors) in enumerate(test_dataloader):
                            output, loss = perform_forward(src_tensors, tgt_tensors)
                            total_loss += loss.item()
                            bleu += calculate_bleu_score(output, tgt_tensors)
                            testbar.update()

                test_losses.append(total_loss / len(test_dataloader))
                test_bleu_scores.append(bleu / len(test_dataloader))
                wandb_log(
                    {
                        "test/loss": test_losses[-1],
                        "test/perplexity": np.exp(test_losses[-1]),
                        "test/bleu": test_bleu_scores[-1],
                    }
                )
                print()
                print(f"Test Loss: [{total_loss=:.3f}] {test_losses[-1]:.3f}")
                print(f"Perplexity: {np.exp(test_losses[-1]):.3f}")
                print(f"BLEU Score: {test_bleu_scores[-1] * 100:.3f}")
                print()

        config.update(
            {
                "pad_src_idx": pad_src_idx,
                "pad_tgt_idx": pad_tgt_idx,
            }
        )
        with open(os.path.join(experiment_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        with open(os.path.join(experiment_dir, "train.json"), "w") as f:
            json.dump(
                {
                    "train_losses": train_losses,
                    "test_losses": test_losses,
                    "train_bleu": train_bleu_scores,
                    "test_bleu": test_bleu_scores,
                },
                f,
                indent=4,
            )

        torch.save(
            transformer.state_dict(),
            os.path.join(experiment_dir, "transformer_final.pth"),
        )

        tokenizer_src.save(os.path.join(experiment_dir, "tokenizer_src.json"))
        tokenizer_tgt.save(os.path.join(experiment_dir, "tokenizer_tgt.json"))

    def inference(
        self,
        checkpoint_path: str,
        experiment_name: str,
        input: Union[str, List[str]],
        top_k: int = -1,
        top_p: float = -1.0,
        temperature: float = 1.0,
        sample: bool = False,
        max_length: int = 100,
        device: str = "cuda:0",
    ) -> None:
        r"""Run inference on the trained model.

        Args:
            checkpoint_path (str):
                The path where the model checkpoints are saved.
            experiment_name (str):
                The name of the experiment.
            input (Union[str, List[str]]):
                The input sentence to generate translation for.
            top_k (int, defaults to *-1*):
                The number of top-k tokens to sample from.
            top_p (float, defaults to *-1.0*):
                The nucleus sampling threshold.
            temperature (float, defaults to *1.0*):
                The temperature for sampling.
            sample (bool, defaults to *False*):
                Whether to sample from the distribution or take the argmax.
            max_length (int, defaults to *100*):
                The maximum length of the generated sequence.
        """
        if isinstance(input, str):
            input = [input]

        experiment_dir = os.path.join(checkpoint_path, experiment_name)
        with open(os.path.join(experiment_dir, "config.json"), "r") as f:
            config = json.load(f)
            print(config)

        # read model
        transformer = EncoderDecoderTransformer(
            num_encoder_layers=config["num_encoder_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            src_vocab_size=config["src_vocab_size"],
            tgt_vocab_size=config["tgt_vocab_size"],
            pad_src_idx=config["pad_src_idx"],
            pad_tgt_idx=config["pad_tgt_idx"],
            embedding_dim=config["embedding_dim"],
            query_key_dim=config["query_key_dim"],
            value_dim=config["value_dim"],
            num_heads=config["num_heads"],
            ffn_hidden_dim=config["ffn_hidden_dim"],
            ffn_activation=config["ffn_activation"],
            use_query_bias=config["use_query_bias"],
            use_key_bias=config["use_key_bias"],
            use_value_bias=config["use_value_bias"],
            use_final_linear_bias=config["use_final_linear_bias"],
            use_pffn_bias=config["use_pffn_bias"],
            dropout_rate=config["dropout_rate"],
            max_length=max_length,
        ).to(device=device)

        transformer.load_state_dict(
            torch.load(os.path.join(experiment_dir, f"{experiment_name}_final.pth")),
            strict=False,
        )

        tokenizer_src = Tokenizer.from_file(os.path.join(experiment_dir, "tokenizer_src.json"))
        tokenizer_tgt = Tokenizer.from_file(os.path.join(experiment_dir, "tokenizer_tgt.json"))

        sos_token_idx = tokenizer_src.token_to_id("<sos>")
        eos_token_idx = tokenizer_src.token_to_id("<eos>")

        transformer.eval()
        with torch.no_grad():
            for i, sentence in enumerate(input):
                en_tokens = [sos_token_idx] + tokenizer_src.encode(sentence.lower()).ids + [eos_token_idx]
                src_tensors = torch.tensor([en_tokens], dtype=torch.long).to(device=device)

                de_tokens = [sos_token_idx]
                memory = transformer.encode(src_tensors)

                for _ in range(max_length - 1):
                    tgt_tensors = torch.tensor([de_tokens], dtype=torch.long).to(device=device)
                    logits = transformer.decode(tgt_tensors, memory)
                    logits /= temperature

                    if top_k > 0:
                        v, _ = torch.topk(logits[:, -1, :], top_k, dim=-1)
                        logits[logits < v[:, -1].unsqueeze(0)] = -float("inf")

                    if top_p > 0.0:
                        sorted_logits, sorted_indices = torch.sort(logits[:, -1, :], descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        logits[:, -1, indices_to_remove] = -float("inf")

                    probs = torch.softmax(logits[:, -1, :], dim=-1)

                    if sample:
                        pred_token = torch.multinomial(probs, num_samples=1)
                    else:
                        _, pred_token = torch.topk(probs, k=1, dim=-1)

                    de_tokens.append(pred_token.item())

                    if pred_token.item() == eos_token_idx:
                        break

                output = tokenizer_tgt.decode(de_tokens, skip_special_tokens=False)
                print(f"Input: {sentence}")
                print(f"Output: {output}")
                print(f"Generated token indices: {de_tokens}")
                print()

    def visualize_positional_encoding(
        self,
        embedding_dim: int = 64,
        max_length: int = 64,
        *,
        save: bool = False,
        output_path: str = "pe.png",
    ) -> None:
        r"""Visualize positional encoding used in the paper.

        Args:
            embedding_dim:
                The dimensionality of vector space embeddings (`d_model` in the paper)
            max_length:
                Maximum sequence length of tokens
            save:
                Whether or not to save the plot
            output_path:
                Path to file where plot is to be saved
        """

        position_encoder = PositionalEncoding(embedding_dim, max_length)
        pe: np.ndarray = position_encoder.pe.detach().numpy()

        figsize = (
            min(embedding_dim // 8, 20),
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