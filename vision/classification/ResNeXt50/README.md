<!--- SPDX-License-Identifier: Apache-2.0 -->

# ResNeXt50

## Description

ResNeXt50 is a simple, highly modularized network architecture for image classification, constructed by a template with cardinality = 32 and bottleneck width = 4d and defined by Xie et al. in their [paper](https://arxiv.org/abs/1611.05431).

## Model

|Model          |Download                       |Top-1 accuracy (%) |Top-5 accuracy (%) |
|---------------|:------------------------------|:------------------|:------------------|
|ResNeXt50      |[96 MB](resnext50.onnx)        |81.096             |95.326             |

## Dataset

[ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/).

## References

* **ResNeXt50** model is from the paper titled [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431).
* This onnx model is converted from a pytorch [model](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnext50_32x4d_a1h-0146ab0a.pth), which is pretrained on ImageNet by ©RossWightman with the code in his [repository](https://github.com/rwightman/pytorch-image-models/).

## License

Apache 2.0
