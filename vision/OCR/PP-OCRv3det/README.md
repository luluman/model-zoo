<!--- SPDX-License-Identifier: Apache 2.0 -->

# PP-OCRv3det

## Description

The PP-OCRv3 detection model is an upgrade of the CML (Collaborative Mutual
Learning) collaborative mutual learning text detection distillation strategy in
PP-OCRv2. PP-OCRv3 is further optimized for detecting teacher model and student
model respectively. Among them, when optimizing the teacher model, the PAN
structure LK-PAN with large receptive field and the DML (Deep Mutual Learning)
distillation strategy are proposed. When optimizing the student model, the FPN
structure RSE-FPN with residual attention mechanism is proposed.

It is recommended to use fastdeploy.

## Model

| Model                          | Download                              | Shape(hw) |
| ------------------------------ |:------------------------------------- |:--------- |
| ch-PP-OCRv3-Detection model    | [3.65 MB](ch_PP-OCRv3_det_infer.tar)  | 128 128   |

* Convert to onnx

``` shell
paddle2onnx  --model_dir ch_PP-OCRv3_det_infer \
             --model_filename inference.pdmodel \
             --params_filename inference.pdiparams \
             --opset_version 13 \
             --save_file ch_PP-OCRv3_det_infer.onnx

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
