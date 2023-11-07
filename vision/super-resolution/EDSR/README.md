<!--- SPDX-License-Identifier: Apache-2.0 -->

# EDSR

## Description

Recent research on super-resolution has progressed with
the development of deep convolutional neural networks
(DCNN). In particular, residual learning techniques exhibit
improved performance. In this paper, we develop an enhanced deep super-resolution network (EDSR) with performance exceeding those of current state-of-the-art SR methods. The significant performance improvement of our model
is due to optimization by removing unnecessary modules in
conventional residual networks. The performance is further
improved by expanding the model size while we stabilize
the training procedure. We also propose a new multi-scale
deep super-resolution system (MDSR) and training method,
which can reconstruct high-resolution images of different
upscaling factors in a single model. The proposed methods
show superior performance over the state-of-the-art methods on benchmark datasets and prove its excellence by winning the NTIRE2017 Super-Resolution Challenge.

## Model

| Model            | Download                                  | Shape(hw) |
| ---------------- |:----------------------------------------- |:--------- |
| EDSR_x2       | [155MB](EDSR_x2.onnx) |  48 48  |
| EDSR_x3       | [166MB](EDSR_x3.onnx) |  85 85  |
| EDSR_x4       | [164MB](EDSR_x4.onnx) |  64 64  |
| MDSR_x2       | [24.8MB](MDSR_x2.onnx) |  64 64  |
| MDSR_x3       | [25.5MB](MDSR_x3.onnx) | 128 128  |
| MDSR_x4       | [25.4MB](MDSR_x4.onnx) | 128 128  |


## Dataset

* [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar)
* [benchmark](https://cv.snu.ac.kr/research/EDSR/benchmark.tar)

## References
* [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf)
  Bee Lim ,Sanghyun Son ,Heewon Kim ,Seungjun Nah ,Kyoung Mu Lee (2017)
* [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch/)

## License

Apache 2.0
