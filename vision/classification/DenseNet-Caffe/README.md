<!--- SPDX-License-Identifier: BSD 3-Clause -->

# DenseNet

## Description

The work on DenseNet won the Best Paper Award of CVPR 2017, the model has much smaller parameters than some SOTA classification networks such as ResNets and Highway Networks, and has a smaller classification error rate.

## Model

|Model          |Download                                                               |Top-1 accuracy (%) |Top-5 accuracy (%) |
|---------------|:----------------------------------------------------------------------|:------------------|:------------------|
| DenseNet      | [model](DenseNet_121.prototxt) [weight](DenseNet_121.caffemodel)      | 74.91             | 92.19             |

## Dataset

[ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/).

## References

* **DenseNet** Model is from the paper [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
* This caffe model is from the [repository](https://github.com/shicai/DenseNet-Caffe) and converted from the original [torch model](https://github.com/liuzhuang13/DenseNet)

## License

BSD 3-Clause
