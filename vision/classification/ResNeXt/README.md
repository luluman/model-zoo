<!--- SPDX-License-Identifier: Apache-2.0 -->

# ResNeXt

## Description

ResNeXt is a classification network further developed after ResNet, which combines stacking blocks and split-transform-merge.

## Model

|Model            |Download                                       |Top-1 accuracy (%) |Top-5 accuracy (%) |
|-----------------|:----------------------------------------------|:------------------|:------------------|
| ResNeXt         | [340M](resnext101_32x8d_traced.pt)            | 79.316            | 94.518            |

## Dataset

[ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/).

## References

* **ResNeXt** model was described in the paper titled [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
* The pre-train model is from the Project: [pytorch-image-models](https://github.com/rwightman/pytorch-image-models).

## License

Apache 2.0
