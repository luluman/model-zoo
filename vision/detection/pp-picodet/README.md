<!--- SPDX-License-Identifier: Apache-2.0 -->

# PicoDet-S

## Description

Baidu developed a series of lightweight models, named PP-PicoDet. Because of the
excellent performance, our models are very suitable for deployment on mobile or
CPU. For more details, please refer to our [report on arXiv](https://arxiv.org/abs/2111.00902).
[website](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/picodet)

It is recommended to use fastdeploy.

## Model

|Model                |Download                              |Shape(hw)     |mAP(0.5:0.95)      |
|---------------------|:-------------------------------------|:-------------|:------------------|
|PicoDet-S            |[4.8 MB](pp_picodet_s.onnx)           |416 416       |32.5               |

* Convert to onnx

``` shell
paddle2onnx  --model_dir . \
             --model_filename model.pdmodel \
             --params_filename model.pdiparams \
             --opset_version 13 \
             --save_file pp_picodet_s.onnx
```

## Dataset

[COCO 2017 dataset](http://cocodataset.org) by Microsoft.

## References

* [PicoDet-S](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_s_320_coco_lcnet_non_postprocess.tar)

## License

Apache-2.0
