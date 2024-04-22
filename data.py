# import inspect
# import json
# import os
# from collections import Counter
# from itertools import combinations
# from typing import Dict, List

# import spacy
# import torch
# from torch.utils.data import Dataset, DataLoader
# from tokenizers import Tokenizer
# from tokenizers.models import BPE
# from tokenizers.trainers import BpeTrainer
# from tokenizers.pre_tokenizers import Whitespace

# from utils import collate_fn


# class Vocabulary:
#     def __init__(self, ignorecase: bool = False) -> None:
#         self.stoi = {}
#         self.itos = {}

#     def add_word(self, word):
#         if word not in self.stoi:
#             index = len(self.stoi)
#             self.stoi[word] = index
#             self.itos[index] = word

#     def __len__(self):
#         return len(self.stoi)


# class Multi30kCharacterTokenizer:
#     def __init__(self) -> None:
#         pass

#     def tokenize_en(self, sentence: str):
#         return list(sentence)

#     def tokenize_de(self, sentence: str):
#         return list(sentence)

#     def decode_en(self, tokens: List[str]):
#         return "".join(tokens)

#     def decode_de(self, tokens: List[str]):
#         return "".join(tokens)


# class Multi30kWordTokenizer:
#     def __init__(self) -> None:
#         pass

#     def tokenize_en(self, sentence: str):
#         return sentence.split(" ")

#     def tokenize_de(self, sentence: str):
#         return sentence.split(" ")

#     def decode_en(self, tokens: List[str]):
#         return " ".join(tokens)

#     def decode_de(self, tokens: List[str]):
#         return " ".join(tokens)


# class Multi30kSpacyTokenizer:
#     def __init__(self) -> None:
#         self.spacy_en = spacy.load("en_core_web_sm")
#         self.spacy_de = spacy.load("de_core_news_sm")

#     def tokenize_en(self, sentence: str):
#         return [tok.text for tok in self.spacy_en.tokenizer(sentence)]

#     def tokenize_de(self, sentence: str):
#         return [tok.text for tok in self.spacy_de.tokenizer(sentence)]

#     def decode_en(self, tokens: List[str]):
#         return " ".join(tokens)

#     def decode_de(self, tokens: List[str]):
#         return " ".join(tokens)


# class Multi30kBPETokenizer:
#     def __init__(
#         self,
#         vocab_size: int = 3000,
#         min_frequency: int = 2,
#         special_tokens: List[str] = [],
#         unk_token: str = "<unk>",
#     ) -> None:
#         self.tokenizer_en = Tokenizer(BPE(unk_token=unk_token))
#         self.tokenizer_de = Tokenizer(BPE(unk_token=unk_token))
#         self.trainer_en = BpeTrainer(
#             special_tokens=special_tokens,
#             min_frequency=min_frequency,
#             vocab_size=vocab_size,
#         )
#         self.trainer_de = BpeTrainer(
#             special_tokens=special_tokens,
#             min_frequency=min_frequency,
#             vocab_size=vocab_size,
#         )
#         self.tokenizer_en.pre_tokenizer = Whitespace()
#         self.tokenizer_de.pre_tokenizer = Whitespace()

#     def train(self, sentences_en: List[str], sentences_de: List[str]) -> None:
#         self.tokenizer_en.train_from_iterator(sentences_en, self.trainer_en)
#         self.tokenizer_de.train_from_iterator(sentences_de, self.trainer_de)

#     def tokenize_en(self, sentence: str):
#         return self.tokenizer_en.encode(sentence).tokens

#     def tokenize_de(self, sentence: str):
#         return self.tokenizer_de.encode(sentence).tokens

#     def decode_en(self, tokens: List[str]):
#         return "".join(tokens)

#     def decode_de(self, tokens: List[str]):
#         return "".join(tokens)


# class Multi30kTokenizer:
#     def __init__(self, tokenizer_type: str, **kwargs) -> None:
#         if tokenizer_type == "word":
#             tokenizer_cls = Multi30kWordTokenizer
#         elif tokenizer_type == "character":
#             tokenizer_cls = Multi30kCharacterTokenizer
#         elif tokenizer_type == "spacy":
#             tokenizer_cls = Multi30kSpacyTokenizer
#         elif tokenizer_type == "bpe":
#             tokenizer_cls = Multi30kBPETokenizer
#         else:
#             raise ValueError(f"Tokenizer type '{tokenizer_type}' is not supported")

#         init_kwargs = {}

#         for kwarg in kwargs:
#             if kwarg in inspect.signature(tokenizer_cls).parameters:
#                 init_kwargs[kwarg] = kwargs[kwarg]

#         self._tok_instance = tokenizer_cls(**init_kwargs)

#     def tokenize_en(self, sentence: str):
#         return self._tok_instance.tokenize_en(sentence)

#     def tokenize_de(self, sentence: str):
#         return self._tok_instance.tokenize_de(sentence)

#     def decode_en(self, tokens: List[str]):
#         return self._tok_instance.decode_en(tokens)

#     def decode_de(self, tokens: List[str]):
#         return self._tok_instance.decode_de(tokens)


# class Multi30kDataset(Dataset):
#     def __init__(
#         self,
#         data: List[Dict[str, str]],
#         sos_token: str,
#         eos_token: str,
#         unk_token: str,
#         pad_token: str,
#         max_length: int,
#         tokenizer: Multi30kTokenizer,
#         en_vocab: Vocabulary,
#         de_vocab: Vocabulary,
#         *args,
#         **kwargs,
#     ) -> None:
#         super().__init__(*args, **kwargs)
#         self.data = data
#         self.sos_token = sos_token
#         self.eos_token = eos_token
#         self.unk_token = unk_token
#         self.pad_token = pad_token
#         self.max_length = max_length
#         self.tokenizer = tokenizer
#         self.en_vocab = en_vocab
#         self.de_vocab = de_vocab

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         en_sentence = item["en"]
#         de_sentence = item["de"]

#         en_tokens = (
#             [self.sos_token]
#             + self.tokenizer.tokenize_en(en_sentence)
#             + [self.eos_token]
#         )[: self.max_length - 2]
#         de_tokens = (
#             [self.sos_token]
#             + self.tokenizer.tokenize_de(de_sentence)
#             + [self.eos_token]
#         )[: self.max_length - 2]

#         en_tensor = torch.tensor(
#             [
#                 self.en_vocab.stoi.get(token, self.en_vocab.stoi[self.unk_token])
#                 for token in en_tokens
#             ],
#             dtype=torch.long,
#         )
#         de_tensor = torch.tensor(
#             [
#                 self.de_vocab.stoi.get(token, self.de_vocab.stoi[self.unk_token])
#                 for token in de_tokens
#             ],
#             dtype=torch.long,
#         )

#         return en_tensor, de_tensor


# class Multi30kDatasetHandler:
#     def __init__(
#         self,
#         path: str,
#         files: Dict[str, str],
#         sos_token: str,
#         eos_token: str,
#         unk_token: str,
#         pad_token: str,
#         max_length: int,
#         tokenizer_type: str,
#         tokenizer_kwargs: Dict[str, str],
#         ignorecase: bool,
#         *args,
#         **kwargs,
#     ) -> None:
#         super().__init__(*args, **kwargs)
#         self.path = path
#         self.files = files
#         self.sos_token = sos_token
#         self.eos_token = eos_token
#         self.unk_token = unk_token
#         self.pad_token = pad_token
#         self.max_length = max_length
#         self.data = {}
#         self.ignorecase = ignorecase
#         self.en_vocab = Vocabulary()
#         self.de_vocab = Vocabulary()

#         tokenizer_kwargs.update({
#             "special_tokens": [self.sos_token, self.eos_token, self.unk_token, self.pad_token],
#             "unk_token": self.unk_token,
#         })
#         self.tokenizer = Multi30kTokenizer(tokenizer_type, **tokenizer_kwargs)

#         for token in [self.sos_token, self.eos_token, self.unk_token, self.pad_token]:
#             self.en_vocab.add_word(token)
#             self.de_vocab.add_word(token)

#         self.load_data()
#         self.build_vocab()

#     def load_data(self):
#         for split, filename in self.files.items():
#             if split not in ["train", "val", "test"]:
#                 raise ValueError(f"Split '{split}' is not supported")
#             self.data[split] = []
#             with open(os.path.join(self.path, filename), "r") as f:
#                 for line in f:
#                     item = json.loads(line)
#                     if self.ignorecase:
#                         item["en"] = item["en"].lower()
#                         item["de"] = item["de"].lower()
#                     self.data[split].append(item)

#     def build_vocab(self):
#         if isinstance(self.tokenizer._tok_instance, Multi30kBPETokenizer):
#             sentences_en = [
#                 item["en"] for split in self.data.keys() for item in self.data[split]
#             ]
#             sentences_de = [
#                 item["de"] for split in self.data.keys() for item in self.data[split]
#             ]

#             self.tokenizer._tok_instance.train(sentences_en, sentences_de)
        
#         for split in self.data.keys():
#             for item in self.data[split]:
#                 for token in self.tokenizer.tokenize_en(item["en"]):
#                     self.en_vocab.add_word(token)
#                 for token in self.tokenizer.tokenize_de(item["de"]):
#                     self.de_vocab.add_word(token)

#     def get_train_dataset(self) -> Multi30kDataset:
#         return Multi30kDataset(
#             data=self.data["train"],
#             sos_token=self.sos_token,
#             eos_token=self.eos_token,
#             unk_token=self.unk_token,
#             pad_token=self.pad_token,
#             max_length=self.max_length,
#             tokenizer=self.tokenizer,
#             en_vocab=self.en_vocab,
#             de_vocab=self.de_vocab,
#         )

#     def get_test_dataset(self) -> Multi30kDataset:
#         return Multi30kDataset(
#             data=self.data["test"],
#             sos_token=self.sos_token,
#             eos_token=self.eos_token,
#             unk_token=self.unk_token,
#             pad_token=self.pad_token,
#             max_length=self.max_length,
#             tokenizer=self.tokenizer,
#             en_vocab=self.en_vocab,
#             de_vocab=self.de_vocab,
#         )

#     def get_val_dataset(self) -> Multi30kDataset:
#         return Multi30kDataset(
#             data=self.data["val"],
#             sos_token=self.sos_token,
#             eos_token=self.eos_token,
#             unk_token=self.unk_token,
#             pad_token=self.pad_token,
#             max_length=self.max_length,
#             tokenizer=self.tokenizer,
#             en_vocab=self.en_vocab,
#             de_vocab=self.de_vocab,
#         )

#     def get_datasets(self):
#         return (
#             self.get_train_dataset(),
#             self.get_test_dataset(),
#             self.get_val_dataset(),
#         )
