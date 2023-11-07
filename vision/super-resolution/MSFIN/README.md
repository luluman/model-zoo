<!--- SPDX-License-Identifier: Apache-2.0 -->

# MSFIN

## Description

As most deep learning methods improve the image super-resolution performance at the cost of higher memory consumption, this research present a lightweight multi-scale feature interaction network (MSFIN). For lightweight SISR, MSFIN expands the receptive field and adequately exploits the informative features of the lowresolution observed images from various scales and interactive connections. In addition, a lightweight recurrent residual channel attention block (RRCAB) is designed so that the network can benefit from the channel attention mechanism while being sufficiently lightweight. Extensive experiments on some benchmarks have confirmed that MSFIN can achieve comparable performance against the stateof-the-arts with a more lightweight model.

## Model

| Model            | Download                                  | Shape(hw) |
| ---------------- |:----------------------------------------- |:--------- |
| MSFIN_x4       | [2.8MB](MSFIN_x4.onnx) | 84 84  |
| MSFIN-S_x4       | [1.4MB](MSFIN-S_x4.onnx) | 84 84  |

## Dataset

* [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar)
* [benchmark](https://cv.snu.ac.kr/research/EDSR/benchmark.tar)

## References
* [Lightweight Image Super-Resolution with Multi-Scale Feature Interaction Network](https://ieeexplore.ieee.org/abstract/document/9428136)
  Z. Wang, G. Gao, J. Li, Y. Yu and H. Lu, "Lightweight Image Super-Resolution with Multi-Scale Feature Interaction Network," 2021 IEEE International Conference on Multimedia and Expo (ICME), Shenzhen, China, 2021, pp. 1-6, doi: 10.1109/ICME51207.2021.9428136.
* [MSFIN：Lightweight Image Super-Resolution with Multi-Scale Feature Interaction Network](https://github.com/wzx0826/MSFIN)

## License

Apache 2.0
