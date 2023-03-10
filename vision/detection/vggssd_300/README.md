<!--- SPDX-License-Identifier: BSD 2-Clause -->

# SSD

## Description

SSD is a target detection algorithm proposed by Wei Liu in 2016. It has obvious speed advantages over Faster RCNN and obvious mAP advantages over YOLO V1.

## Model

|Model      |Download                                                                              | mAP               |
|-----------|:-------------------------------------------------------------------------------------|:------------------|
|SSD        |[model](deploy.prototxt) [weight](VGG_coco_SSD_300x300_iter_400000.caffemodel)        | 25.1              |

## Dataset

* Training data: [COCO trainval35k](https://github.com/insikk/coco_dataset_trainval35k).
* Test data: [COCO test-dev2015](https://github.com/cocodataset/cocoapi/issues/87).

## References

* [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
* [weiliu89/caffe](https://github.com/weiliu89/caffe/tree/ssd)

## License

BSD 2-Clause
