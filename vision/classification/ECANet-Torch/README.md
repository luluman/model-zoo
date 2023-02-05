<!--- SPDX-License-Identifier: MIT -->

# ECANet

## Description

ECANet is a type of convolutional neural network that utilises an Efficient Channel Attention module. The authors propose an Efficient Channel Attention ( ECA ) module for deep CNN, which avoids dimensionality reduction and captures cross-channel interactions in an effective way.

## Model

|Model            |Download                             |Top-1 accuracy (%) |Top-5 accuracy (%) |
|-----------------|:------------------------------------|:------------------|:------------------|
| ECANet          | [99M](eca_resnet50_k3557_trace.pt)  | 77.42             | 93.62             |

## Dataset

[ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/).

## References

* **ECANet** is from the paper [ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks](https://arxiv.org/abs/1910.03151)
* This pre-train model is from the [BangguWu/ECANet](https://github.com/BangguWu/ECANet)

## License

MIT
