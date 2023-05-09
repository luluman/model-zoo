<!--- SPDX-License-Identifier: GPL-3.0 -->

# PP-HumanSeg

## Description

Human segmentation is a high-frequency application in the field of image segmentation.
Generally, human segentation can be classified as portrait segmentation and general human segmentation.

For portrait segmentation and general human segmentation, PaddleSeg releases the PP-HumanSeg models, which has **good performance in accuracy, inference speed and robustness**. Besides, we can deploy PP-HumanSeg models to products without training
Besides, PP-HumanSeg models can be deployed to products at zero cost, and it also support fine-tuning to achieve better performance.

It is recommended to use fastdeploy.

## Model
| Model                      | Download                                                         | Shape(hw) | mIou(%) |
| ---------------------------|:-----------------------------------------------------------------|:--------- |:--------|
| pp-humansegv1-lite-softmax | [580.7 KB](human_pp_humansegv1.zip)                              | 192 192   | 86.02   |
| pp-humansegv1-lite-argmax  | [580.6 KB](human_pp_humansegv1_lite_192x192_inference_model.zip) | 192 192   | 86.02   |

## Dataset

* [mini\_supervisely](https://paddleseg.bj.bcebos.com/humanseg/data/mini_supervisely.zip)

## References

* [PP-HumanSeg: Connectivity-Aware Portrait Segmentation with a Large-Scale Teleconferencing Video Dataset](https://arxiv.org/abs/2112.07146)
* [PaddlePaddle/PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/contrib/PP-HumanSeg)

## License

GPL 3.0
