<!--- SPDX-License-Identifier: MIT -->

# R-FCN

## Description

R-FCN is a type of region-based object detector. The whole network is mainly composed of full convolution network ( ResNet ) and RPN network. The former is used to extract features, and the latter is used to generate ROI.

## Model

|Model      |Download                                                                              | mAP               |
|-----------|:-------------------------------------------------------------------------------------|:------------------|
|R-FCN      |[model](test_agnostic_del_py_layer.prototxt) [weight](resnet101_rfcn_coco.caffemodel) | 27.9              |

## Dataset

* [COCO 2014 train](http://images.cocodataset.org/zips/train2014.zip).
* [COCO 2014 val](http://images.cocodataset.org/zips/val2014.zip).

## References

* [R-FCN: Object Detection via Region-based Fully Convolutional Networks](https://arxiv.org/abs/1605.06409)
* [YuwenXiong/py-R-FCN](https://github.com/YuwenXiong/py-R-FCN)

## License

MIT
