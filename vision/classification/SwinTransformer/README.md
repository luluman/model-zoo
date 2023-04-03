<!--- SPDX-License-Identifier: Apache-2.0 -->

# Swin Transformer

## Description

Swin Transformer is a transformer-based general-purpose backbone for computer
vision tasks, which use shifted window scheme and introduced by Ze at al. in
their paper in 2021.

## Model

|Model          |Download                       |Top-1 accuracy (%) |Top-5 accuracy (%) |
|---------------|:------------------------------|:------------------|:------------------|
|Swin-Tiny      |[28 MB](swin_t.onnx)           |80.9               |96.0               |

## Dataset

[ImageNet (ILSVRC2012)](<http://www.image-net.org/challenges/LSVRC/2012/>).

## References

* **Swin Transformer** model is from the paper titled
[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030).
* This onnx model is converted from a pytorch
[model](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth),
which is pretrained on ImageNet-22K with the code in his
[repository](https://github.com/rwightman/pytorch-image-models/).

## License

Apache 2.0
