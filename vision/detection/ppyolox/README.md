<!--- SPDX-License-Identifier: Apache-2.0 -->

# ppyolov3

## Description

YOLOX: Exceeding YOLO Series in 2021.

It is recommended to use fastdeploy.

## Model

| Model                         | Download                                   | Shape(hw) |
| ----------------------------- |:------------------------------------------ |:--------- |
| yolox_s_300e_coco             | [35MB](pp_yolox_s_coco.onnx)               | [640 640] |

* Convert to onnx

``` shell
python3 paddle_infer_shape.py  \
        --model_dir . \
        --model_filename model.pdmodel \
        --params_filename model.pdiparams \
        --save_dir new_model \
        --input_shape_dict="{'image':[1,3,640,640],'scale_factor':[1,2]}"

paddle2onnx  --model_dir new_model \
             --model_filename model.pdmodel \
             --params_filename model.pdiparams \
             --opset_version 13 \
             --save_file pp_yolovx_s_coco.onnx
```

## Dataset

* [coco](http://images.cocodataset.org/zips/val2017.zip)

## References

* [YOLOX](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/yolox)
* [PaddleDetection](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/paddledetection)
* [yolox_s_300e_coco](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/yolox)

## License

Apache 2.0
