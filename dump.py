# --- ScaledDotProductAttention --- 

# import torch

# from models import ScaledDotProductAttention

# model = ScaledDotProductAttention(query_key_size=512)
# query = key = torch.randn((1, 128, 512, 64))
# value = torch.randn((1, 128, 512, 64))

# output = model(query, key, value)
# print(output.shape)

# assert output.shape == (1, 128, 512, 64)



# --- MultiHeadAttention --- 

# import torch

# from models import MultiHeadAttention

# model = MultiHeadAttention(
#     embedding_size=512,
#     query_key_size=64,
#     value_size=64,
#     num_heads=8,
# )

# query = key = torch.randn((1, 128, 512))
# value = torch.randn((1, 128, 512))

# output = model(query, key, value)
# print(output.shape)

# assert output.shape == (1, 128, 512)



# --- PositionWiseFFN --- 

# import torch

# from models import MultiHeadAttention, PositionWiseFFN

# model = MultiHeadAttention(
#     embedding_size=512,
#     query_key_size=64,
#     value_size=64,
#     num_heads=8,
# )
# ffn = PositionWiseFFN(
#     embedding_size=512,
#     hidden_size=128, # 2048, for testing
#     activation="relu"
# )

# query = key = torch.randn((1, 128, 512))
# value = torch.randn((1, 128, 512))

# output = model(query, key, value)
# output = ffn(output)
# print(output.shape)

# assert output.shape == (1, 128, 512)



# --- EncoderBlock & DecoderBlock --- 

# import torch

# from models import DecoderBlock, EncoderBlock

# enc = EncoderBlock(
#     embedding_size=512,
#     query_key_size=64,
#     value_size=64,
#     num_heads=8,
#     ffn_hidden_dim=128, # 2048
# )
# dec = DecoderBlock(
#     embedding_size=512,
#     query_key_size=64,
#     value_size=64,
#     num_heads=8,
#     ffn_hidden_dim=128, # 2048
# )

# x = torch.randn((1, 128, 512))
# y = torch.randn((1, 128, 512))
# enc_output = enc(x)
# dec_output = dec(y, x)

# print(enc_output.shape)
# print(dec_output.shape)

# assert enc_output.shape == (1, 128, 512)
# assert dec_output.shape == (1, 128, 512)



# --- EncoderBlock & DecoderBlock --- 

import torch

from models import Transformer


def get_information(model: torch.nn.Module) -> dict:
    trainable_params = sum([param.numel() for param in model.parameters() if param.requires_grad])
    non_trainable_params = sum([param.numel() for param in model.parameters() if not param.requires_grad])

    return {
        "trainable": trainable_params,
        "non-trainable": non_trainable_params,
    }

# 66 million
model = Transformer(
    num_layers=6,
    vocab_src_size=25000,
    vocab_tgt_size=25000,
    pad_src_idx=24999,
    pad_tgt_idx=24999,
    embedding_size=512,
    query_key_size=64,
    value_size=64,
    num_heads=8,
    ffn_hidden_dim=2048,
)

print(model)

print(get_information(model))

src_tokens = torch.randint(0, 25000, (1, 128))
tgt_tokens = torch.randint(0, 25000, (1, 128))

output = model(src_tokens, tgt_tokens)

print(output.shape)

assert output.shape == (1, 128, 25000)
