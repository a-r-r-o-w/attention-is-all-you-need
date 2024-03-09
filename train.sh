#!/bin/bash

python3 main.py train \
  --num_layers=6 \
  --vocab_src_size=-1 \
  --vocab_tgt_size=-1 \
  --pad_src_idx=-1 \
  --pad_tgt_idx=-1 \
  --embedding_size=512 \
  --query_key_size=64 \
  --value_size=64 \
  --num_heads=8 \
  --ffn_hidden_dim=2048 \
  --ffn_activation="relu" \
  --use_ffn_bias_1 \
  --use_ffn_bias_2 \
  --dropout_rate=0.1 \
  --max_length=32 \
  --weight_initialization_method="kaiming_uniform" \
  --learning_rate=1e-04 \
  --weight_decay=0.0001 \
  --batch_size=32 \
  --dataset_name="multi30k" \
  --tokenizer_type="spacy" \
  --epochs=100 \
  --seed=42
