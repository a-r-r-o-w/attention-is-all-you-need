import copy
import subprocess
import traceback
from typing import Any, Dict


def run_experiment(config: Dict[str, Any]) -> None:
    try:
        command = [
            "python3",
            "main.py",
            "train",
        ]
        for key, value in config.items():
            if isinstance(value, bool):
                if value:
                    command.append(f"--{key}")
            else:
                command.append(f"--{key}={value}")
        subprocess.run(command, check=True)
    except Exception as e:
        traceback.print_exc()
        print(f"Exception: {e}")


def experiment_batch_size(config: Dict[str, Any]) -> None:
    for batch_size in [16, 32, 64, 128, 256, 512, 1024]:
        config["batch_size"] = batch_size
        config["experiment_name"] = f"transformer_batch_size_{batch_size}"
        run_experiment(config)


basic_config = {
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "vocab_src_size": 5000,
    "vocab_tgt_size": 5000,
    "embedding_dim": 512,
    "query_key_dim": 512,
    "value_dim": 512,
    "num_heads": 8,
    "ffn_hidden_dim": 1024,
    "ffn_activation": "relu",
    "use_pffn_bias": True,
    "dropout_rate": 0.1,
    "max_length": 32,
    "weight_initialization_method": "kaiming_uniform",
    "learning_rate": 1e-4,
    "weight_decay": 0.0001,
    "batch_size": 32,
    "dataset_name": "multi30k",
    "epochs": 20,
    "seed": 42,
    "validation_epochs": 1,
    "checkpoint_path": "checkpoints",
    "experiment_name": "transformer",
    "checkpoint_steps": 10000,
    "gradient_accumulation_steps": 1,
    "track_wandb": True,
}

experiment_batch_size(copy.deepcopy(basic_config))
