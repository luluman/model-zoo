<!--- SPDX-License-Identifier: MIT -->

# Xception

## Description

Xception is another improvement of Inception_v3 proposed by Google after Inception. The author thinks that it is better to deal with the correlation of channels and the correlation of space separately, so separable convolution is used to replace the convolution operation in Inception_v3.

## Model

|Model           |Download                                                         |Top-1 accuracy (%) |Top-5 accuracy (%) |
|----------------|:----------------------------------------------------------------|:------------------|:------------------|
| Xception       |[model](deploy_xception.prototxt) [weight](xception.caffemodel)  | 79.1              | 94.51             |

## Dataset

[ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/).

## References

* **Xception** was described in the paper titled [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
* The pre-train model is from the [repository](https://github.com/soeaver/caffe-model/tree/master/cls) and converted from the Project: [keras deep-learning-models](https://github.com/fchollet/deep-learning-models)

## License

MIT
