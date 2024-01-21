# Attention Is All You Need

A nearly faithful implementation of the reknowned "Attention Is All You Need" paper.

- Paper: https://arxiv.org/abs/1706.03762
- Original Codebase: https://github.com/tensorflow/tensor2tensor

I've been trying to practice implementing papers related to new model architectures for the past few months, in the hope to become more confident with large codebases and new ideas. In the past, I've implemented this paper and its variant architectures a few times for learning purposes. This was my first attempt at implementing a paper directly by referring to the architecture, ofcourse with the insights from past memory.

I've recorded myself walking through the paper and implementing it in a two hour programming stream on [YouTube](https://youtu.be/Fu1oZdYYQYE). This is an attempt to make myself more comfortable with explaining ideas when programming. Unfortunately, I own a terrible microphone, only realizing that after completing the implementation, and so the audio has been replaced with lofi.

### TODO

- [ ] Verify model architecture to be faithful with the original paper
- [ ] Verify if all hyperparameters are set to correct defaults
- [ ] Verify if layernorm and dropouts are correct
- [ ] Implement the training loop
- [ ] Implement BLEU score metric
- [ ] Play around with different hyperparameters and try to really understand things
- [ ] Implement Attention maps for visualization
- [ ] Improve documentation
- [ ] Visualize positional encoding. Compare against learned embeddings for position as suggested in paper
- [ ] Allow different number of encoder-decoder blocks. Experiment with decoder-only model as done in a few architectures
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
