<!--- SPDX-License-Identifier: Apache-2.0 -->

# ppyolov3

## Description

The PaddlePaddle version of YOLOv3.

It is recommended to use fastdeploy.

## Model

| Model                         | Download                                   | Shape(hw) |
| ----------------------------- |:------------------------------------------ |:--------- |
| yolov3_mobilenet_v3_270e_coco | [91MB](yolov3_mobilenet_v3_270e_coco.onnx) | 320 320   |

* Convert to onnx

``` shell
python3 paddle_infer_shape.py \
        --model_dir . \
        --model_filename model.pdmodel \
        --params_filename model.pdiparams \
        --save_dir new_model \
        --input_shape_dict \
        "{'scale_factor':[1,2],'image':[1,3,320,320], 'im_shape':[1,2]}"

paddle2onnx  --model_dir new_model \
             --model_filename model.pdmodel \
             --params_filename model.pdiparams \
             --opset_version 13 \
             --save_file yolov3_mobilenet_v3_270e_coco.onnx
```

## Dataset

* [coco](http://images.cocodataset.org/zips/val2017.zip)

## References

* [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
* [YOLOv3](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/yolov3)
* [PaddleDetection](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/paddledetection)

## License

Apache 2.0
