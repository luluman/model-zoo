<!--- SPDX-License-Identifier: Apache 2.0 -->

# PP-OCRv3rec

## Description

The recognition module of PP-OCRv3 is optimized based on the text recognition
algorithm SVTR. RNN is abandoned in SVTR, and the context information of the
text line image is more effectively mined by introducing the Transformers
structure, thereby improving the text recognition ability.

It is recommended to use fastdeploy.

[website](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/PP-OCRv3_introduction_en.md)

## Model

| Model                          | Download                              | Shape(hw) |
| ------------------------------ |:------------------------------------- |:--------- |
| ch-PP-OCRv3-Recognition model  | [12 MB](ch_PP-OCRv3_rec_infer.tar)    | 48 320    |

* Convert to onnx

``` shell
paddle2onnx  --model_dir ch_PP-OCRv3_rec_infer \
             --model_filename inference.pdmodel \
             --params_filename inference.pdiparams \
             --opset_version 13 \
             --save_file ch_PP-OCRv3_rec_infer.onnx
```

## Dataset

* [ICDAR 2019-LSVT](https://aistudio.baidu.com/aistudio/datasetdetail/177210)
* [ICDAR 2017-RCTW-17](https://rctw.vlrlab.net/dataset)
* [中文街景文字识别](https://aistudio.baidu.com/aistudio/datasetdetail/8429)
* [ICDAR 2019-ArT](https://ai.baidu.com/broad/download?dataset=art)

## References

* [PP-OCRv3: More Attempts for the Improvement of Ultra Lightweight OCR System](https://arxiv.org/abs/2206.03001v2)
* [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/ocr)

## License

Apache 2.0
