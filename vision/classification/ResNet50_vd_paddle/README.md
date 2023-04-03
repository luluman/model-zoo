<!--- SPDX-License-Identifier: Apache-2.0 -->

# ResNet50-vd

## Description

The ResNet series model was proposed in 2015 and won the championship in the ILSVRC2015 competition with a top5 error rate of 3.57%. The network innovatively proposed the residual structure, and built the ResNet network by stacking multiple residual structures. Experiments show that using residual blocks can improve the convergence speed and accuracy effectively.

It is recommended to use fastdeploy.

## Model

|Model          |Download                                                                 |ONNX version   |Opset version  |Top-1 accuracy (%) |Top-5 accuracy (%) |
|---------------|:------------------------------------------------------------------------|:--------------|:--------------|:------------------|:------------------|
|ResNet50-vd    |[98 MB](https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz)|v8             |7              |79.12              |94.44              |

## Dataset

[ImageNet (ILSVRC2015)](http://www.image-net.org/challenges/LSVRC/2015/).

## References

* [PaddlePaddle/PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/ResNet_and_vd.md)

## License

Apache 2.0
