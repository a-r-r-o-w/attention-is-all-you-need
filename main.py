import os
import json
from typing import List, Optional, Union

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import Transformer, PositionalEncoding, LRScheduler
from utils import get_summary, initialize_weights, collate_fn


def _print_with_line(content: str, line_length: int = 80):
    print(content)
    print("-" * line_length)


def greedy_decode(
    transformer: Transformer,
    src: torch.Tensor,
    max_length: int,
    sos_token_idx: int,
    eos_token_idx: int
) -> torch.Tensor:
    encoded = transformer.encode(src)
    outputs = torch.ones(1, 1).fill_(sos_token_idx).type_as(src).to(src.device)
    
    for _ in range(max_length - 1):
        output = transformer.decode(outputs, encoded)
        output = output.argmax(dim=-1)
        pred_token = output[0, -1].item()
        outputs = torch.cat([outputs, torch.ones(1, 1).fill_(pred_token).type_as(src)], dim=-1)
        if pred_token == eos_token_idx:
            break
    
    # print(outputs)
    return outputs


def seq_to_seq_translate(
    transformer: Transformer,
    en_tensors: torch.Tensor,
    de_tensors: torch.Tensor,
    tokenizer_en: Tokenizer,
    tokenizer_de: Tokenizer,
    sos_token: str,
    eos_token: str,
    pad_token: str,
    max_length: int,
):
    en_tensors = en_tensors.to(device="cuda")
    de_tensors = de_tensors.to(device="cuda")
    current_batch_size = en_tensors.shape[0]
    sos_token_idx = tokenizer_de.token_to_id(sos_token)
    eos_token_idx = tokenizer_de.token_to_id(eos_token)
    pad_token_idx = tokenizer_de.token_to_id(pad_token)
    outputs = []
    
    for i in range(current_batch_size):
        src = en_tensors[i].unsqueeze(0)
        tgt = de_tensors[i].unsqueeze(0)
        output = greedy_decode(transformer, src, max_length, sos_token_idx, eos_token_idx)[:max_length]
        if output.size(1) < max_length:
            output = torch.cat([output, torch.ones(1, max_length - output.size(1)).fill_(pad_token_idx).type_as(src)], dim=-1)
        outputs.append(output)
    
    outputs = torch.cat(outputs, dim=0)

    targets = tokenizer_de.decode_batch(de_tensors.cpu().numpy(), skip_special_tokens=False)
    generated = tokenizer_de.decode_batch(outputs.cpu().numpy(), skip_special_tokens=False)

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
        dropout_rate: float = 0.1,
        max_length: int = 10000,
        weight_initialization_method: str = "kaiming_uniform",
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        dataset_name: str = "multi30k",
        epochs: int = 10,
        seed: int = 42,
        validation_epochs: int = 1,
        checkpoint_path: str = "checkpoints",
        experiment_name: str = "transformer",
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

        sos_token = "<sos>"
        eos_token = "<eos>"
        unk_token = "<unk>"
        pad_token = "<pad>"

        experiment_dir = os.path.join(checkpoint_path, experiment_name)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir, exist_ok=True)

        match dataset_name:
            case "multi30k":
                path = "dataset/multi30k"
                files = {
                    "train": "train.jsonl",
                    "test": "test.jsonl",
                    "val": "val.jsonl",
                }
                data = {}
                for split, filename in files.items():
                    if split not in ["train", "val", "test"]:
                        raise ValueError(f"Split '{split}' is not supported")
                    
                    data[split] = []

                    with open(os.path.join(path, filename), "r") as f:
                        for line in f:
                            item = json.loads(line)
                            item["en"] = item["en"].lower()
                            item["de"] = item["de"].lower()
                            data[split].append(item)
                    
                    # data[split] = data[split][:10000]

                sentences_en = [item["en"] for split in data.keys() for item in data[split]]
                sentences_de = [item["de"] for split in data.keys() for item in data[split]]

                tokenizer_en = Tokenizer(BPE(unk_token=unk_token))
                tokenizer_de = Tokenizer(BPE(unk_token=unk_token))
                tokenizer_en.pre_tokenizer = Whitespace()
                tokenizer_de.pre_tokenizer = Whitespace()

                trainer_en = BpeTrainer(special_tokens=[sos_token, eos_token, unk_token, pad_token], vocab_size=vocab_src_size, min_frequency=2)
                trainer_de = BpeTrainer(special_tokens=[sos_token, eos_token, unk_token, pad_token], vocab_size=vocab_tgt_size, min_frequency=2)

                tokenizer_en.train_from_iterator(sentences_en, trainer_en)
                tokenizer_de.train_from_iterator(sentences_de, trainer_de)
            case _:
                raise ValueError(f"Dataset {dataset_name} not supported")
        
        sos_token_idx = tokenizer_en.token_to_id(sos_token)
        eos_token_idx = tokenizer_en.token_to_id(eos_token)
        for split in data.keys():
            data_tensors = []
            for item in data[split]:
                item["en"] = [sos_token_idx] + tokenizer_en.encode(item["en"]).ids + [eos_token_idx]
                item["de"] = [sos_token_idx] + tokenizer_de.encode(item["de"]).ids + [eos_token_idx]
                item["en"] = torch.tensor(item["en"][:max_length], dtype=torch.long)
                item["de"] = torch.tensor(item["de"][:max_length], dtype=torch.long)
                data_tensors.append(item)
            data[split] = data_tensors
        
        if pad_src_idx == -1:
            pad_src_idx = tokenizer_en.token_to_id(pad_token)
        if pad_tgt_idx == -1:
            pad_tgt_idx = tokenizer_de.token_to_id(pad_token)

        def collate_helper(batch):
            return collate_fn(
                batch,
                en_pad_token_id=pad_src_idx,
                de_pad_token_id=pad_tgt_idx,
                max_length=max_length,
            )
        
        train_dataloader = DataLoader(
            [(item["en"], item["de"]) for item in data["train"]],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_helper,
        )
        val_dataloader = DataLoader(
            [(item["en"], item["de"]) for item in data["val"]],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_helper,
        )
        test_dataloader = DataLoader(
            [(item["en"], item["de"]) for item in data["test"]],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_helper,
        )

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
            dropout_rate=dropout_rate,
            max_length=max_length,
        ).to(device="cuda")

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

        # plt.ion()
        # fig, (ax1, ax2) = plt.subplots(2, 1)
        # ax1.set_xlabel("Epoch")
        # ax1.set_ylabel("Loss")
        # ax1.set_title("Training and Validation Loss")
        # (line1,) = ax1.plot([], [], "r-", label="Train loss")  # line for train loss
        # (line2,) = ax1.plot([], [], "g-", label="Validation loss")  # line for validation loss
        # ax1.legend()

        # ax2.set_xlabel("Step")
        # ax2.set_ylabel("Learning Rate")
        # ax2.set_title("Learning Rate")
        # (line3,) = ax2.plot([], [], "b-")  # line for learning rate

        with tqdm(total=total_steps, desc="Training") as train_bar:
            for epoch in range(1, epochs + 1):
                total_loss = 0.0

                transformer.train()
                for i, (en_tensors, de_tensors) in enumerate(train_dataloader):
                    en_tensors = en_tensors.to(device="cuda")
                    de_tensors = de_tensors.to(device="cuda")
                    src_de = de_tensors[:, :-1]
                    tgt_de = de_tensors[:, 1:].contiguous().view(-1)
                    
                    optimizer.zero_grad()
                    output = transformer(en_tensors, src_de)
                    loss = criterion(output.contiguous().view(-1, vocab_tgt_size), tgt_de)

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
                    train_bar.update()

                    if step % checkpoint_steps == 0:
                        torch.save(
                            transformer.state_dict(),
                            os.path.join(
                                experiment_dir, f"{experiment_name}_{step}.pth"
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
                                src_de = de_tensors[:, :-1]
                                tgt_de = de_tensors[:, 1:].contiguous().view(-1)

                                output = transformer(en_tensors, src_de)
                                loss = criterion(output.contiguous().view(-1, vocab_tgt_size), tgt_de)
                                total_loss += loss.item()
                                valbar.update()

                    val_losses.append(total_loss / len(val_dataloader))
                    print()
                    print(
                        f"Validation Loss: [{total_loss=:.3f}] {total_loss / len(val_dataloader):.3f}"
                    )
                    print(f"Perplexity: {np.exp(total_loss / len(val_dataloader)):.3f}")
                    print()

                    print("Running inference on validation set")
                    tgt_tokens = tokenizer_de.decode_batch(
                        de_tensors[:5, 1:].cpu().numpy(), skip_special_tokens=False
                    )
                    output_tokens = tokenizer_de.decode_batch(
                        output[:5].argmax(dim=-1).cpu().numpy(), skip_special_tokens=False
                    )

                    for tgt, out in zip(tgt_tokens, output_tokens):
                        print(f"   target: {tgt}")
                        print(f"generated: {out}")
                        print()

                # transformer.eval()
                # with torch.no_grad():
                #     examples = 5
                #     for i, (en_tensors, de_tensors) in enumerate(test_dataloader):
                #         en_tensors = en_tensors[:examples]
                #         de_tensors = de_tensors[:examples]
                #         targets, generated = seq_to_seq_translate(
                #             transformer,
                #             en_tensors,
                #             de_tensors,
                #             tokenizer_en,
                #             tokenizer_de,
                #             sos_token,
                #             eos_token,
                #             pad_token,
                #             max_length,
                #         )
                #         print("Running testset inference")
                #         for target, gen in zip(targets, generated):
                #             print(f"   target: {target}")
                #             print(f"generated: {gen}")
                #             print()
                #         break

                # line1.set_xdata(range(1, len(train_losses) + 1))
                # line1.set_ydata(train_losses)
                # line2.set_xdata(range(1, len(val_losses) + 1))
                # line2.set_ydata(val_losses)
                # ax1.relim()
                # ax1.autoscale_view()

                # line3.set_xdata(range(1, len(learning_rates) + 1))
                # line3.set_ydata(learning_rates)
                # ax2.relim()
                # ax2.autoscale_view()

                # # Redraw the figure
                # fig.canvas.draw()
                # fig.canvas.flush_events()

        # plt.ioff()
        # plt.show()
        
        with open(os.path.join(experiment_dir, "config.json"), "w") as f:
            json.dump(
                {
                    "num_encoder_layers": num_encoder_layers,
                    "num_decoder_layers": num_decoder_layers,
                    "vocab_src_size": vocab_src_size,
                    "vocab_tgt_size": vocab_tgt_size,
                    "pad_src_idx": pad_src_idx,
                    "pad_tgt_idx": pad_tgt_idx,
                    "embedding_size": embedding_size,
                    "query_key_size": query_key_size,
                    "value_size": value_size,
                    "num_heads": num_heads,
                    "ffn_hidden_dim": ffn_hidden_dim,
                    "ffn_activation": ffn_activation,
                    "use_query_bias": use_query_bias,
                    "use_key_bias": use_key_bias,
                    "use_value_bias": use_value_bias,
                    "use_ffn_bias_1": use_ffn_bias_1,
                    "use_ffn_bias_2": use_ffn_bias_2,
                    "use_final_linear_bias": use_final_linear_bias,
                    "dropout_rate": dropout_rate,
                    "max_length": max_length,
                    "weight_initialization_method": weight_initialization_method,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "dataset_name": dataset_name,
                    "epochs": epochs,
                    "seed": seed,
                    "validation_epochs": validation_epochs,
                    "checkpoint_path": checkpoint_path,
                    "experiment_name": experiment_name,
                    "checkpoint_steps": checkpoint_steps,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                },
                f,
                indent=4,
            )
        
        with open(os.path.join(experiment_dir, f"train.json"), "w") as f:
            json.dump(
                {
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "learning_rates": learning_rates,
                },
                f,
                indent=4,
            )
        
        torch.save(
            transformer.state_dict(),
            os.path.join(experiment_dir, "transformer_final.pth"),
        )

        tokenizer_en.save(os.path.join(experiment_dir, "tokenizer_en.json"))
        tokenizer_de.save(os.path.join(experiment_dir, "tokenizer_de.json"))
    
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
    ) -> None:
        if isinstance(input, str):
            input = [input]
        
        experiment_dir = os.path.join(checkpoint_path, experiment_name)
        with open(os.path.join(experiment_dir, "config.json"), "r") as f:
            config = json.load(f)
        
        # read model
        transformer = Transformer(
            num_encoder_layers=config["num_encoder_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            vocab_src_size=config["vocab_src_size"],
            vocab_tgt_size=config["vocab_tgt_size"],
            pad_src_idx=config["pad_src_idx"],
            pad_tgt_idx=config["pad_tgt_idx"],
            embedding_size=config["embedding_size"],
            query_key_size=config["query_key_size"],
            value_size=config["value_size"],
            num_heads=config["num_heads"],
            ffn_hidden_dim=config["ffn_hidden_dim"],
            ffn_activation=config["ffn_activation"],
            use_query_bias=config["use_query_bias"],
            use_key_bias=config["use_key_bias"],
            use_value_bias=config["use_value_bias"],
            use_ffn_bias_1=config["use_ffn_bias_1"],
            use_ffn_bias_2=config["use_ffn_bias_2"],
            use_final_linear_bias=config["use_final_linear_bias"],
            dropout_rate=config["dropout_rate"],
            max_length=max_length,
        ).to(device="cuda")

        transformer.load_state_dict(
            torch.load(os.path.join(experiment_dir, f"{experiment_name}_final.pth")), strict=False
        )

        tokenizer_en = Tokenizer.from_file(os.path.join(experiment_dir, "tokenizer_en.json"))
        tokenizer_de = Tokenizer.from_file(os.path.join(experiment_dir, "tokenizer_de.json"))

        sos_token_idx = tokenizer_en.token_to_id("<sos>")
        eos_token_idx = tokenizer_en.token_to_id("<eos>")

        transformer.eval()
        with torch.no_grad():
            for i, sentence in enumerate(input):
                en_tokens = [sos_token_idx] + tokenizer_en.encode(sentence.lower()).ids + [eos_token_idx]
                en_tensors = torch.tensor([en_tokens], dtype=torch.long).to(device="cuda")

                de_tokens = [sos_token_idx]
                memory = transformer.encode(en_tensors)

                for _ in range(max_length - 1):
                    de_tensors = torch.tensor([de_tokens], dtype=torch.long).to(device="cuda")
                    logits = transformer.decode(de_tensors, memory)
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

                output = tokenizer_de.decode(de_tokens, skip_special_tokens=False)
                print(f"Input: {sentence}")
                print(f"Output: {output}")
                print(f"Generated token indices: {de_tokens}")
                print()

        # transformer.eval()
        # with torch.no_grad():
        #     for i, sentence in enumerate(input):
        #         en_tensors = [sos_token_idx] + tokenizer_en.encode(sentence.lower()).ids + [eos_token_idx]
        #         en_tensors = torch.tensor([en_tensors], dtype=torch.long).to(device="cuda")

        #         de_tokens = [sos_token_idx]
        #         encoded = transformer.encode(en_tensors)

        #         for _ in range(config["max_length"] - 1):
        #             de_tensors = torch.tensor([de_tokens], dtype=torch.long).to(device="cuda")
        #             logits = transformer.decode(en_tensors, de_tensors, encoded)

        #             if top_k > 0:
        #                 v, _ = torch.topk(logits[:, -1, :], top_k, dim=-1)
        #                 logits[logits < v[:, -1].unsqueeze(0)] = -float("inf")
                    
        #             probs = logits.softmax(dim=2)
        #             _, idx_next = torch.topk(probs, k=1, dim=2)
        #             print(idx_next.shape)
        #             pred_token = idx_next[0, -1, 0].item()
        #             de_tokens.append(pred_token)
                    
        #             if pred_token == eos_token_idx:
        #                 break
                
        #         output = tokenizer_de.decode(de_tokens, skip_special_tokens=False)
        #         print(f"Input: {sentence}")
        #         print(f"Output: {output}")
        #         print()


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
