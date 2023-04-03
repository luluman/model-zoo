<!--- SPDX-License-Identifier: Apache-2.0 -->

# Resnet50-v1

## Description

Deeper neural networks are more difficult to train. Residual
learning framework ease the training of networks that are
substantially deeper. The research explicitly reformulate
the layers as learning residual functions with reference to
the layer inputs, instead of learning unreferenced functions.
ResNet models perform image classification - they take
images as input and classify the major object in the image
into a set of pre-defined classes. They are trained on
ImageNet dataset which contains images from 1000 classes.
ResNet models provide very high accuracies with affordable
model sizes. They are ideal for cases when high accuracy of
classification is required.

## Model

Resnet50-v1-7

|Model        |ONNX version |Opset version|Top-1 accuracy (%)|Top-5 accuracy (%)|
|-------------|:------------|:------------|:-----------------|:-----------------|
|Resnet50     |1.2.1        |7            |74.93             |92.38             |

## Dataset

* Dataset used for train and validation: [ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/).

## References

* [Deep residual learning for image recognition](https://arxiv.org/abs/1512.03385)
* [onnx/models/vision/classification/resnet/](https://github.com/onnx/models/tree/main/vision/classification/resnet)

## License

Apache 2.0
