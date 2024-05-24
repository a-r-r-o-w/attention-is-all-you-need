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


def experiment_activation(config: Dict[str, Any]) -> None:
    for activation in [
        "sigmoid",
        "relu",
        "gelu",
        "silu",
        "glu",
        "reglu",
        "geglu",
        "swiglu",
        "tanh",
        "elu",
        "leaky_relu",
    ]:
        config["ffn_activation"] = activation
        config["experiment_name"] = f"transformer_activation_{activation}"
        run_experiment(config)


def experiment_learning_rate(config: Dict[str, Any]) -> None:
    for learning_rate in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5, 1e-6]:
        config["learning_rate"] = learning_rate
        config["experiment_name"] = f"transformer_learning_rate_{learning_rate}"
        run_experiment(config)


def experiment_num_heads(config: Dict[str, Any]) -> None:
    for num_heads in [1, 2, 4, 8, 16, 32]:
        config["num_heads"] = num_heads
        config["experiment_name"] = f"transformer_num_heads_{num_heads}"
        run_experiment(config)


def experiment_num_encoder_decoder_layers(config: Dict[str, Any]) -> None:
    for num_encoder_layers in [1, 2, 3]:
        for num_decoder_layers in [1, 2, 3]:
            config["num_encoder_layers"] = num_encoder_layers
            config["num_decoder_layers"] = num_decoder_layers
            config["experiment_name"] = (
                f"transformer_num_encoder_decoder_layers_{num_encoder_layers}_{num_decoder_layers}"
            )
            run_experiment(config)

    config["num_encoder_layers"] = 6
    config["num_decoder_layers"] = 6
    config["experiment_name"] = "transformer_num_encoder_decoder_layers_6_6"
    run_experiment(config)


def experiment_ffn_hidden_dim(config: Dict[str, Any]) -> None:
    for ffn_hidden_dim in [128, 256, 512, 1024, 2048]:
        config["ffn_hidden_dim"] = ffn_hidden_dim
        config["experiment_name"] = f"transformer_ffn_hidden_dim_{ffn_hidden_dim}"
        run_experiment(config)


basic_config = {
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "src_vocab_size": 5000,
    "tgt_vocab_size": 5000,
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
    "device": "cuda:0",
    "track_wandb": True,
}

# experiment_batch_size(copy.deepcopy(basic_config))
# experiment_activation(copy.deepcopy(basic_config))
# experiment_learning_rate(copy.deepcopy(basic_config))
# experiment_num_heads(copy.deepcopy(basic_config))
# experiment_num_encoder_decoder_layers(copy.deepcopy(basic_config))
# experiment_ffn_hidden_dim(copy.deepcopy(basic_config))
