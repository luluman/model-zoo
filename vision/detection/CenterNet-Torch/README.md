<!--- SPDX-License-Identifier: MIT -->

# CenterNet

## Description

CenterNet is an end-to-end object detection model based on free-anchor. It inherits from the CornerNet model and can be easily migrated to tasks such as 3D object detection and human key point detection.

## Model

|Model              |Download                                       | AP (no augmentation / flip augmentation / multi scale augmentation)  |
|-------------------|:----------------------------------------------|:---------------------------------------------------------------------|
|CenterNet(pytorch) |[71M](ctdet_coco_dlav0_1x.torchscript.pt)      | 36.3 / 38.2 / 40.7                                                   |

## Dataset

[COCO 2017 dataset](http://cocodataset.org)

## References

* [Objects as Points](https://arxiv.org/abs/1904.07850)
* [xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet)

## License

MIT
