<!--- SPDX-License-Identifier: Apache-2.0 -->

# EfficientNet-B1

## Description

EfficientNet-B1 is one of the EfficientNet family of models, which are convolutional neural network architecture and scaling methods that uniformly scales all dimensions of depth/width/resolution using a compound coefficient.

## Model

|Model            |Download                         |#Params (M)        |#FLOPs (B)         |Top-1 accuracy (%) |Top-5 accuracy (%) |
|-----------------|:--------------------------------|:------------------|:------------------|:------------------|:------------------|
| EfficientNet-B1 | [32MB](efficientnet-b1.onnx)    | 7.8               | 0.70              | 79.1              | 94.4              |

## Dataset

[ImageNet](https://image-net.org/)

## References

* **EfficientNet-B1** is from the paper titled [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946).
* This pre-train onnx model is converted from a [pytorch model](https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth), which is from the [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch).

## License

Apache 2.0
