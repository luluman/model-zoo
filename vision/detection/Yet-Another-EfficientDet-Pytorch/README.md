<!--- SPDX-License-Identifier: LGPL-3.0 -->

# EfficientDet

## Description

EfficientDet is a target detection algorithm proposed by Google Brain in 2019, which adopts EfficientNet as the backbone feature extraction network, it has eight different structures from D0 to D7.

## Model

|Model              |Download                                          |Shape(hw)          |FPS                |mAP                |
|-------------------|:-------------------------------------------------|:------------------|:------------------|:------------------|
|EfficientDet       |[model](efficientdet-d0_trace.pt)                 |512 512            |36.20              |33.1               |

## Dataset

[COCO 2017 dataset](http://cocodataset.org)

## References

* [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)
* [zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)

## License

LGPL-3.0 license
