<!--- SPDX-License-Identifier: Apache-2.0 -->

# EfficientNet-B7

## Description

EfficientNet-B7 is one of the EfficientNet family of models, which are convolutional neural network architecture and scaling methods that uniformly scales all dimensions of depth/width/resolution using a compound coefficient. It achieves state-of-the-art accuracy on ImageNet with 66M parameters and 37B FLOPS, being 8.4x smaller and 6.1x faster on CPU inference than (previous best) Gpipe.

## Model

|Model            |Download                         |#Params (M)        |#FLOPs (B)         |Top-1 accuracy (%) |Top-5 accuracy (%) |
|-----------------|:--------------------------------|:------------------|:------------------|:------------------|:------------------|
| EfficientNet-B7 | [257MB](efficientnet-b7.onnx)   | 66                | 37                | 84.3              | 97.0              |

## Dataset

[ImageNet](https://image-net.org/)

## References

* **EfficientNet-B7** is from the paper titled [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946).
* This pre-train onnx model is converted from a [pytorch model](https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth), which is from the [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch).

## License

Apache 2.0
