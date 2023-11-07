<!--- SPDX-License-Identifier: Apache-2.0 -->

# MSRN

## Description

Blindly increasing the depth of the network cannot ameliorate the network effectively, and may greatly increase the requirements for training tricks. This research propose a novel multiscale residual network (MSRN) to fully exploit the image features, in which Multi-scale Residual Block (MSRB) is applied. Based on the residual block, MSRB introduces convolution kernels of different sizes to adaptively detect the image features in different scales, and then lets these features interact with each other to get the most efficacious image information. Finally, all these features are fused and sent to the reconstruction module for recovering the high-quality image.

## Model

| Model            | Download                                  | Shape(hw) |
| ---------------- |:----------------------------------------- |:--------- |
| MSRN_x2       | [23.2MB](MSRN_x2.onnx) | 100 100  |
| MSRN_x3       | [23.9MB](MSRN_x3.onnx) | 100 100  |
| MSRN_x4       | [23.8MB](MSRN_x4.onnx) | 100 100  |
| MSRN_x8       | [24.3MB](MSRN_x8.onnx) | 100 100  |

## Dataset

* [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar)
* [benchmark](https://cv.snu.ac.kr/research/EDSR/benchmark.tar)

## References
* [Multi-scale Residual Network for Image Super-Resolution](https://openaccess.thecvf.com/content_ECCV_2018/papers/Juncheng_Li_Multi-scale_Residual_Network_ECCV_2018_paper.pdf)
  Li, J., Fang, F., Mei, K., Zhang, G. (2018)
* [MSRN_PyTorch](https://github.com/MIVRC/MSRN-PyTorch)

## License

Apache 2.0
