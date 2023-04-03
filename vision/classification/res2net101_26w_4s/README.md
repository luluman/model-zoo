<!--- Creative Commons Attribution-NonCommercial-ShareAlike 4.0 -->

# Res2net101

## Description

The Res2Net represents multi-scale features at a granular level
and increases the range of receptive fields for each network layer.
The proposed Res2Net block can be plugged into the state-of-the-art backbone
CNN models,
e.g., ResNet, ResNeXt, and DLA.
We evaluate the Res2Net block on all these models and demonstrate consistent
performance gains over baseline models on widely-used datasets.

## Model

Res2net101

|Model        |Params                                 |MACCs        |top-1 error       |top-5 error       |
|-------------|:--------------------------------------|:------------|:-----------------|:-----------------|
|Res2net101   |[45.21M](res2net101_26w_4s.onnx)       |8.1          |20.81             |5.57              |

## Dataset

* Dataset used for train and validation: [ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/).

## References

* [IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)](https://arxiv.org/pdf/1904.01169.pdf)
* [Res2Net/Res2Net-PretrainedModels](https://github.com/Res2Net/Res2Net-PretrainedModels)

## License

The code is released under the Creative Commons
Attribution-NonCommercial-ShareAlike
4.0 International Public License for Noncommercial use only.
Any commercial use should get formal permission first.
