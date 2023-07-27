# Extending ViLBERT for Vision-and-Language Navigation (VLN)

The architecture of VLN-BERT is structurally similar to ViLBERT. This is by
design, because it enables straightforward transfer of visual grounding
learned from large-scale web data to the embodied task of VLN.

Specifically, we make a small number of VLN-specific changes to ViLBERT that
are structured as augmentations (adding modules) rather than ablations
(removing existing network components) so that pretrained weights can be
transferred to initialize large portions of the model.

All of the modifications to ViLBERT are annotated in the file `vilbert.py`
with the comment:

```python
# Note: modified for vln >
...
# Note: modified for vln <
```

## Code Origins

The files in this folder were adapted from a variety of sources. Here are their origin stories:

- `vilbert.py` is modified from
[https://github.com/jiasenlu/vilbert_beta](https://github.com/jiasenlu/vilbert_beta)

- `optimization.py` is adapted from
[https://github.com/huggingface/transformers/blob/1.2.0/pytorch_transformers/optimization.py](https://github.com/huggingface/transformers/blob/1.2.0/pytorch_transformers/optimization.py)

- `file_utils.py` is adapted from
[https://github.com/huggingface/transformers/blob/0.1.2/pytorch_pretrained_bert/file_utils.py](https://github.com/huggingface/transformers/blob/0.1.2/pytorch_pretrained_bert/file_utils.py),
which was originally adapted from
[https://github.com/allenai/allennlp](https://github.com/allenai/allennlp)
