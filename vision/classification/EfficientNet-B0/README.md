<!--- SPDX-License-Identifier: Apache-2.0 -->

# EfficientNet-B0

## Description

EfficientNet-B0 was introduced by Tan et al. in 2019. Starting from the baseline EfficientNet-B0, they applied compound scaling method to scale it up to obtain EfficientNet-B1 to B7, which achieve much better accuracy and efficiency than previous ConvNets.

## Model

|Model            |Download                         |#Params (M)        |#FLOPs (B)         |Top-1 accuracy (%) |Top-5 accuracy (%) |
|-----------------|:--------------------------------|:------------------|:------------------|:------------------|:------------------|
| EfficientNet-B0 | [22MB](efficientnet-b0.onnx)    | 5.3               | 0.39              | 77.1              | 93.3              |

## Dataset

[ImageNet](https://image-net.org/)

## References

* **EfficientNet-B0** is from the paper titled [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946).
* This pre-train onnx model is converted from a [pytorch model](https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth), which is from the [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch).

## License

Apache 2.0
