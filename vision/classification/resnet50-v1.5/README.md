<!--- SPDX-License-Identifier: Apache License 2.0 -->

# Huggingface-resnet50

## Description

ResNet (Residual Network) is a convolutional neural network that democratized the concepts of residual learning and skip connections. This enables to train much deeper models.

This is ResNet v1.5, which differs from the original model: in the bottleneck blocks which require downsampling, v1 has stride = 2 in the first 1x1 convolution, whereas v1.5 has stride = 2 in the 3x3 convolution. This difference makes ResNet50 v1.5 slightly more accurate (~0.5% top1) than v1, but comes with a small performance drawback (~5% imgs/sec) according to [Nvidia](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch).

## Model

|Model                  |Download                      |
|-----------------------|:-----------------------------|
|huggingface-resnet50   |[97.85 MB](resnet50v1.5.onnx) |


## References

* [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50)

* [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

## License

Apache License 2.0
