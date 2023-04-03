<!--- SPDX-License-Identifier: GPL-3.0 -->

# PP-OCRv3cls

## Description

The direction classifier is used in the scene where the image is not 0 degrees.
In this scene, it is necessary to perform a correction operation on the text
line detected in the picture. In the PaddleOCR system, The text line image
obtained after text detection is sent to the recognition model after affine
transformation.

At this time, only a 0 and 180 degree angle classification of the text is
required, so the built-in PaddleOCR text angle classifier only supports 0 and
180 degree classification.

It is recommended to use fastdeploy.

## Model

| Model                            | Download                                      | Shape(hw) |
| -------------------------------- |:--------------------------------------------- |:--------- |
| ch-PP-OCRv3-Direction classifier | [2.08 MB](ch_ppocr_mobile_v2.0_cls_infer.tar) | 128 128   |

* Convert to onnx

``` shell
paddle2onnx  --model_dir ch_ppocr_mobile_v2.0_cls_infer \
             --model_filename inference.pdmodel \
             --params_filename inference.pdiparams \
             --opset_version 13 \
             --save_file ch_ppocr_mobile_v2.0_cls_infer.onnx
```

## Dataset

* [ICDAR 2019-LSVT](https://aistudio.baidu.com/aistudio/datasetdetail/177210)
* [ICDAR 2017-RCTW-17](https://rctw.vlrlab.net/dataset)
* [中文街景文字识别](https://aistudio.baidu.com/aistudio/datasetdetail/8429)
* [ICDAR 2019-ArT](https://ai.baidu.com/broad/download?dataset=art)

## References

* [PP-OCRv3: More Attempts for the Improvement of Ultra Lightweight OCR System](https://arxiv.org/abs/2206.03001v2)
* [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.5)

## License

Apache 2.0
