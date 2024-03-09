#!/bin/sh

dataset1_folder=iitb_english_hindi
dataset2_folder=multi30k

mkdir -p $dataset1_folder
wget -O $dataset1_folder/train.parquet https://huggingface.co/datasets/cfilt/iitb-english-hindi/resolve/main/data/train-00000-of-00001.parquet
wget -O $dataset1_folder/test.parquet https://huggingface.co/datasets/cfilt/iitb-english-hindi/resolve/main/data/test-00000-of-00001.parquet
wget -O $dataset1_folder/validate.parquet https://huggingface.co/datasets/cfilt/iitb-english-hindi/resolve/main/data/validation-00000-of-00001.parquet

mkdir -p $dataset2_folder
wget -O $dataset2_folder/train.jsonl https://huggingface.co/datasets/bentrevett/multi30k/raw/main/train.jsonl
wget -O $dataset2_folder/test.jsonl https://huggingface.co/datasets/bentrevett/multi30k/raw/main/test.jsonl
wget -O $dataset2_folder/val.jsonl https://huggingface.co/datasets/bentrevett/multi30k/raw/main/val.jsonl

