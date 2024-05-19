#!/bin/bash

python3 main.py train \
  --num_encoder_layers=3 \
  --num_decoder_layers=3 \
  --src_vocab_size=5000 \
  --tgt_vocab_size=5000 \
  --embedding_dim=256 \
  --query_key_dim=256 \
  --value_dim=256 \
  --num_heads=8 \
  --ffn_hidden_dim=512 \
  --ffn_activation="swiglu" \
  --use_pffn_bias \
  --dropout_rate=0.1 \
  --max_length=32 \
  --weight_initialization_method="kaiming_uniform" \
  --learning_rate=2e-4 \
  --weight_decay=0.0001 \
  --batch_size=32 \
  --dataset_name="multi30k" \
  --epochs=20 \
  --seed=42 \
  --validation_epochs=1 \
  --checkpoint_path="checkpoints" \
  --experiment_name="transformer" \
  --checkpoint_steps=10000 \
  --gradient_accumulation_steps=1 \
  --device="cuda:0" \
  --track_wandb
