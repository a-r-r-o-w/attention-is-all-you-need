# Attention Is All You Need

A nearly faithful implementation of the reknowned "Attention Is All You Need" paper.

- Paper: https://arxiv.org/abs/1706.03762
- Original Codebase: https://github.com/tensorflow/tensor2tensor

I've been trying to practice implementing papers related to new model architectures for the past few months, in the hope to become more confident with large codebases and new ideas. In the past, I've implemented this paper and its variant architectures a few times for learning purposes. This was my first attempt at implementing a paper directly by referring to the architecture, ofcourse with the insights from past memory.

I've recorded myself walking through the paper and implementing it in a two hour programming stream on [YouTube](https://youtu.be/Fu1oZdYYQYE). This is an attempt to make myself more comfortable with explaining ideas when programming. Unfortunately, I own a terrible microphone, only realizing that after completing the implementation, and so the audio has been replaced with lofi.

### Usage

#### Training

TODO

#### Visualization of Position Encoding

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

### TODO

- [x] Verify model architecture to be faithful with the original paper
- [x] Verify if all hyperparameters are set to correct defaults
- [x] Verify if layernorm and dropouts are correct
- [x] Implement the training loop
- [x] Implement the LR scheduler
- [x] Implement greedy decoding
- [ ] Implement beam search decoding
- [ ] Implement BLEU score metric
- [ ] Implement Attention maps for visualization
- [ ] Improve documentation
- [x] Visualize positional encoding
- [ ] Compare positional encoding against learned embeddings for position as suggested in paper
- [ ] Improve README and add illustrations where required

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
