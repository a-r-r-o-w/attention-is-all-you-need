#!/bin/bash

python3 main.py train \
  --num_encoder_layers=3 \
  --num_decoder_layers=3 \
  --vocab_src_size=5000 \
  --vocab_tgt_size=5000 \
  --pad_src_idx=-1 \
  --pad_tgt_idx=-1 \
  --embedding_size=512 \
  --query_key_size=512 \
  --value_size=512 \
  --num_heads=8 \
  --ffn_hidden_dim=512 \
  --ffn_activation="relu" \
  --use_pffn_bias \
  --dropout_rate=0.1 \
  --max_length=32 \
  --weight_initialization_method="kaiming_uniform" \
  --learning_rate=1e-4 \
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
