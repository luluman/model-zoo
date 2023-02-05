<!--- SPDX-License-Identifier: CC0 -->

# Inception_v2

## Description

Inception_v2 model is an improvement of the Inception_v1, the improvements mainly include the introduction of Batch-Normalization and the use of two 3 × 3 convolutions instead of 5 × 5.

## Model

|Model          |Download                                                               |Top-1 accuracy (%) |Top-5 accuracy (%) |
|---------------|:----------------------------------------------------------------------|:------------------|:------------------|
| Inception_v2  | [model](Inception21k.prototxt) [weight](Inception21k.caffemodel)      | 68.3              | 89.0              |

## Dataset

* This model is a pretrained model on [full imagenet dataset](https://ieeexplore.ieee.org/document/5206848) with 14,197,087 images in 21,841 classes.
* Dataset used for validation: [ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/).

## References

* **Inception_v2** Model was described in the paper titled [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
* Refer to the [repository](https://github.com/pertusa/InceptionBN-21K-for-Caffe) for more details.

## License

CC0
