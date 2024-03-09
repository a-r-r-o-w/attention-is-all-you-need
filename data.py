import json
import os
from typing import Dict, List

import spacy
import torch
from torch.utils.data import Dataset, DataLoader

from utils import collate_fn


class Vocabulary:
    def __init__(self) -> None:
        self.stoi = {}
        self.itos = {}

    def add_word(self, word):
        if word not in self.stoi:
            index = len(self.stoi)
            self.stoi[word] = index
            self.itos[index] = word

    def __len__(self):
        return len(self.stoi)


class Multi30kCharacterTokenizer:
    def __init__(self) -> None:
        pass

    def tokenize_en(self, sentence: str):
        return list(sentence)

    def tokenize_de(self, sentence: str):
        return list(sentence)


class Multi30kWordTokenizer:
    def __init__(self) -> None:
        pass

    def tokenize_en(self, sentence: str):
        return sentence.split(" ")

    def tokenize_de(self, sentence: str):
        return sentence.split(" ")


class Multi30kSpacyTokenizer:
    def __init__(self) -> None:
        self.spacy_en = spacy.load("en_core_web_sm")
        self.spacy_de = spacy.load("de_core_news_sm")

    def tokenize_en(self, sentence: str):
        return [tok.text for tok in self.spacy_en.tokenizer(sentence)]

    def tokenize_de(self, sentence: str):
        return [tok.text for tok in self.spacy_de.tokenizer(sentence)]


class Multi30kTokenizer:
    def __init__(self, tokenizer_type: str) -> None:
        self.tokenizer = None

        if tokenizer_type == "word":
            self.tokenizer = Multi30kWordTokenizer()
        elif tokenizer_type == "character":
            self.tokenizer = Multi30kCharacterTokenizer()
        elif tokenizer_type == "spacy":
            self.tokenizer = Multi30kSpacyTokenizer()
        else:
            raise ValueError(f"Tokenizer type '{tokenizer_type}' is not supported")

    def tokenize_en(self, sentence: str):
        return self.tokenizer.tokenize_en(sentence)

    def tokenize_de(self, sentence: str):
        return self.tokenizer.tokenize_de(sentence)


class Multi30kDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, str]],
        sos_token: str,
        eos_token: str,
        unk_token: str,
        pad_token: str,
        max_length: int,
        tokenizer: Multi30kTokenizer,
        en_vocab: Vocabulary,
        de_vocab: Vocabulary,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.data = data
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.en_vocab = en_vocab
        self.de_vocab = de_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        en_sentence = item["en"]
        de_sentence = item["de"]

        en_tokens = (
            [self.sos_token]
            + self.tokenizer.tokenize_en(en_sentence)[: self.max_length - 2]
            + [self.eos_token]
        )
        de_tokens = (
            [self.sos_token]
            + self.tokenizer.tokenize_de(de_sentence)[: self.max_length - 2]
            + [self.eos_token]
        )

        en_tensor = torch.tensor(
            [
                self.en_vocab.stoi.get(token, self.en_vocab.stoi[self.unk_token])
                for token in en_tokens
            ],
            dtype=torch.long,
        )
        de_tensor = torch.tensor(
            [
                self.de_vocab.stoi.get(token, self.de_vocab.stoi[self.unk_token])
                for token in de_tokens
            ],
            dtype=torch.long,
        )

        return en_tensor, de_tensor


class Multi30kDatasetHandler:
    def __init__(
        self,
        path: str,
        files: Dict[str, str],
        sos_token: str,
        eos_token: str,
        unk_token: str,
        pad_token: str,
        max_length: int,
        tokenizer_type: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.path = path
        self.files = files
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.max_length = max_length
        self.data = {}
        self.tokenizer = Multi30kTokenizer(tokenizer_type)
        self.en_vocab = Vocabulary()
        self.de_vocab = Vocabulary()

        for token in [self.sos_token, self.eos_token, self.unk_token, self.pad_token]:
            self.en_vocab.add_word(token)
            self.de_vocab.add_word(token)

        self.load_data()
        self.build_vocab()

    def load_data(self):
        for split, filename in self.files.items():
            if split not in ["train", "val", "test"]:
                raise ValueError(f"Split '{split}' is not supported")
            self.data[split] = []
            with open(os.path.join(self.path, filename), "r") as f:
                for line in f:
                    self.data[split].append(json.loads(line))

    def build_vocab(self):
        for split in self.data.keys():
            length = len(self.data[split])
            ten_percent = length // 1
            for item in self.data[split][:ten_percent]:
                for token in self.tokenizer.tokenize_en(item["en"]):
                    self.en_vocab.add_word(token)
                for token in self.tokenizer.tokenize_de(item["de"]):
                    self.de_vocab.add_word(token)

    def get_train_dataset(self) -> Multi30kDataset:
        return Multi30kDataset(
            data=self.data["train"],
            sos_token=self.sos_token,
            eos_token=self.eos_token,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            max_length=self.max_length,
            tokenizer=self.tokenizer,
            en_vocab=self.en_vocab,
            de_vocab=self.de_vocab,
        )

    def get_test_dataset(self) -> Multi30kDataset:
        return Multi30kDataset(
            data=self.data["test"],
            sos_token=self.sos_token,
            eos_token=self.eos_token,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            max_length=self.max_length,
            tokenizer=self.tokenizer,
            en_vocab=self.en_vocab,
            de_vocab=self.de_vocab,
        )

    def get_val_dataset(self) -> Multi30kDataset:
        return Multi30kDataset(
            data=self.data["val"],
            sos_token=self.sos_token,
            eos_token=self.eos_token,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            max_length=self.max_length,
            tokenizer=self.tokenizer,
            en_vocab=self.en_vocab,
            de_vocab=self.de_vocab,
        )

    def get_datasets(self):
        return (
            self.get_train_dataset(),
            self.get_test_dataset(),
            self.get_val_dataset(),
        )
