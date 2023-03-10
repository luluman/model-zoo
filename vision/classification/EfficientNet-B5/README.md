<!--- SPDX-License-Identifier: Apache-2.0 -->

# EfficientNet-B5

## Description

EfficientNet-B5 is one of the EfficientNet family of models, which are convolutional neural network architecture and scaling methods that uniformly scales all dimensions of depth/width/resolution using a compound coefficient.

## Model

|Model            |Download                         |#Params (M)        |#FLOPs (B)         |Top-1 accuracy (%) |Top-5 accuracy (%) |
|-----------------|:--------------------------------|:------------------|:------------------|:------------------|:------------------|
| EfficientNet-B5 | [119MB](efficientnet-b5.onnx)   | 30                | 9.9               | 83.6              | 96.7              |

## Dataset

[ImageNet](https://image-net.org/)

## References

* **EfficientNet-B5** is from the paper titled [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946).
* This pre-train onnx model is converted from a [pytorch model](https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth), which is from the [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch).

## License

Apache 2.0
