<!--- SPDX-License-Identifier: MIT -->

# SiamMask

## Description

SiamMask is a simple real-time method for visual object tracking and semi-supervised video object segmentation. This model is simple, versatile, and fast. Its effect also surpasses other tracking methods. It has achieved competitive performance and speed for the semi-supervised video object segmentation task on DAVIS-2016 and DAVIS-2017.

## Model

|Model                 |Download                              |DAVIS2016 (J / F)     |DAVIS2017 (J / F)    |Speed (FPS)          |
|----------------------|:-------------------------------------|:---------------------|:--------------------|:--------------------|
|SiamMask(pytorch)     |[83MB](SiamMask_DAVIS_trace.pt)       |0.713 / 0.674         |0.543 / 0.585        |56                   |
|SiamMask(onnx)        |[66MB](SiamMask_DAVIS.onnx)           |0.713 / 0.674         |0.543 / 0.585        |56                   |
## Dataset

* Youtube-VOS
* [COCO](http://cocodataset.org/#download)
* [ImageNet-DET](http://image-net.org/challenges/LSVRC/2015/)
* [ImageNet-VID](http://image-net.org/challenges/LSVRC/2015/)

## References

* [Fast Online Object Tracking and Segmentation: A Unifying Approach](https://arxiv.org/abs/1812.05050)
* [foolwood/SiamMask](https://github.com/foolwood/SiamMask)

## License

MIT
