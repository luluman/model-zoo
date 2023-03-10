<!--- SPDX-License-Identifier: Apache-2.0 -->

# VGG19

## Description

VGG19 is a variant of VGG model which consists of 19 layers (16 convolution layers and 3 Fully connected layer).

## Model

|Model          |Download                       |Top-1 accuracy (%) |Top-5 accuracy (%) |
|---------------|:------------------------------|:------------------|:------------------|
|VGG19          |[549 MB](vgg19.onnx)           |72.366             |90.872             |

## Dataset

[ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/).

## References

* **VGG19** model is from the paper titled [Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556).
* This onnx model is converted from a pytorch [model](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth), which is pretrained on ImageNet by ©RossWightman with the code in his [repository](https://github.com/rwightman/pytorch-image-models/).

## License

Apache 2.0
