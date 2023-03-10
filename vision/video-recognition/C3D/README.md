<!--- SPDX-License-Identifier: Apache-2.0 -->

# C3D

## Description

C3D is a modified version of BVLC caffe to support 3-Dimensional Convolutional
Networks for video features extraction.

## Model

Accuracy tested by C3D + linear SVM.

| Model                         | Download                                   | Shape(hw) |  UCF101  |  ASLAN     |  UMD-Scene  |  YUPENN-Scene  |  Object  |
| ----------------------------- |:------------------------------------------ |:--------- |:-------- |:---------- |:----------- |:-------------- |:-------- |
| c3d_traced                    | [300MB](c3d_traced.pt)                     | 112 112   |  82.3    | 78.3(86.5) |  87.7       |  98.1          |  22.3    |

## Dataset

* [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)
* [ASLAN](https://talhassner.github.io/home/projects/ASLAN/ASLAN-main.html)
* [UMD-Scene](https://theclarice.umd.edu/venues/scene-shop)
* [YUPENN-Scene](https://vision.eecs.yorku.ca/research/dynamic-scenes/)
* [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

## References

* [jfzhang95/pytorch-video-recognition](https://github.com/jfzhang95/pytorch-video-recognition)
* [Learning Spatiotemporal Features with 3D Convolutional Networks](https://vlg.cs.dartmouth.edu/c3d/c3d_video.pdf)

## License

Apache 2.0
