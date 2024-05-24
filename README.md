# Attention Is All You Need

A nearly faithful implementation of the reknowned "Attention Is All You Need" paper.

- Paper: https://arxiv.org/abs/1706.03762
- Original Codebase: https://github.com/tensorflow/tensor2tensor

I've been trying to practice implementing papers related to new model architectures for the past few months, in the hope to become more confident with large codebases and new ideas. In the past, I've implemented this paper and its variant architectures a few times for learning purposes. This was my first attempt at implementing a paper directly by referring to the architecture, ofcourse with the insights from past memory.

I've recorded myself walking through the paper and implementing it in a two hour programming stream on [YouTube](https://youtu.be/Fu1oZdYYQYE). This is an attempt to make myself more comfortable with explaining ideas when programming. Unfortunately, I own a terrible microphone, only realizing that after completing the implementation, and so the audio has been replaced with lofi.

### Usage

<details>
<summary> Training </summary>

```py
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
  --checkpoint_path="checkpoints" \
  --experiment_name="transformer" \
  --checkpoint_steps=10000 \
  --gradient_accumulation_steps=1 \
  --device="cuda:0" \
  --track_wandb
```
</details>

<details>

<summary> Visualizing positional encoding </summary>

To run visualization for positional encoding:

```py
python3 main.py visualize_positional_encoding -e 64 -m 64 --save -o assets/pe-64-64.png
```

This should give you the visual representation of the sin and cosine positional encoding based on Section 3.5 of the paper. There are many good resources explaining the need for positional encodings, either learned embeddings or the encoding used here, and I do not think I could do a better job at explaining it. So, here's a few good reads:

- https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
- https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3
- https://www.youtube.com/watch?v=1biZfFLPRSY

<table>
<tr>
  <td><strong> Positional Encoding </strong></td>
  <td><strong> Visualization </strong></td>
</tr>
<tr>
  <td>
    $$\text{{PE}}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
    $$\text{{PE}}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$
  </td>
  <td><img src="https://github.com/a-r-r-o-w/attention-is-all-you-need/blob/main/assets/pe-64-64.png"></td>
</tr>
</table>

</details>

### References

- https://peterbloem.nl/blog/transformers
- https://jalammar.github.io/illustrated-transformer/
- https://github.com/tensorflow/tensor2tensor
- https://github.com/huggingface/transformers
- https://www.youtube.com/watch?v=iDulhoQ2pro
- https://www.youtube.com/watch?v=cbYxHkgkSVs

```
@misc{vaswani2023attention,
    title={Attention Is All You Need}, 
    author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
    year={2023},
    eprint={1706.03762},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
