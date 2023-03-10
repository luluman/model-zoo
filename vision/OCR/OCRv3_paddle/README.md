<!--- SPDX-License-Identifier: Apache-2.0 -->

# PP-OCRv3

## Description

PP-OCRv3 was proposed by baidu, it contains a series of tasks with multiple models, including
- Text detection `DBDetector`
- [Optional] Direction classification `Classifer` is used to adjust the direction of images before text recognition
- Character recognition `Recognizer` is used to recognize characters from images  

Paddle2ONNX needs to be installed for the conversion of model
```shell
#1. change the Paddle model from dynamic inputs to fixed inputs, see paddle_infer_shape.py
#det model
python3 paddle_infer_shape.py --model_dir ch_PP-OCRv3_det_infer \
                             --model_filename inference.pdmodel \
                             --params_filename inference.pdiparams \
                             --save_dir ch_PP-OCRv3_det_infer \
                             --input_shape_dict="{'x':[1,3,960,608]}"
#cls model
python3 paddle_infer_shape.py --model_dir ch_ppocr_mobile_v2.0_cls_infer \
                             --model_filename inference.pdmodel \
                             --params_filename inference.pdiparams \
                             --save_dir ch_ppocr_mobile_v2.0_cls_infer \
                             --input_shape_dict="{'x':[1,3,48,192]}"
#rec model
python3 paddle_infer_shape.py --model_dir ch_PP-OCRv3_rec_infer \
                             --model_filename inference.pdmodel \
                             --params_filename inference.pdiparams \
                             --save_dir ch_PP-OCRv3_rec_infer \
                             --input_shape_dict="{'x':[1,3,48,584]}"

#2. Convert the Paddle model with fixed inputs to the ONNX model, see paddle2onnx
#det model 
paddle2onnx --model_dir ch_PP-OCRv3_det_infer \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file ch_PP-OCRv3_det_infer_fix.onnx \
            --enable_dev_version True
#cls model
paddle2onnx --model_dir ch_ppocr_mobile_v2.0_cls_infer \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file ch_ppocr_mobile_v2.0_cls_infer.onnx \
            --enable_dev_version True
#rec model
paddle2onnx --model_dir ch_PP-OCRv3_rec_infer \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file ch_PP-OCRv3_rec_infer.onnx \
            --enable_dev_version True
```
## Model

|Model                      |Download                                                                                      |
|---------------------------|:---------------------------------------------------------------------------------------------|
| ch_PP-OCRv3_det           | [3.6 MB](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar)         |
| ch_ppocr_mobile_v2.0_cls  | [2.1 MB](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) |
| ch_PP-OCRv3_rec           | [11.3 MB](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar)        |

## References
* [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md)

## License

Apache 2.0
