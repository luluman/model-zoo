<!--- SPDX-License-Identifier: Apache-2.0 -->

# ppyoloe

## Description

PP-YOLOE is an excellent single-stage anchor-free model based on PP-YOLOv2,
surpassing a variety of popular YOLO models. PP-YOLOE has a series of models,
named s/m/l/x, which are configured through width multiplier and depth multiplier.
PP-YOLOE avoids using special operators, such as Deformable Convolution or
Matrix NMS, to be deployed friendly on various hardware.

It is recommended to use fastdeploy.

## Model

| Model                       | Download                                 | Shape(hw) |
| --------------------------- |:---------------------------------------- |:--------- |
| ppyoloe_crn_x_300e_coco     | [341MB](ppyoloe_crn_x_300e_coco.zip)     | 640 640   |
| ppyoloe_crn_s_300e_coco     | [31MB](ppyoloe_crn_s_300e_coco.onnx)     | 640 640   |
| ppyoloe_plus_crn_x_80e_coco | [345MB](ppyoloe_plus_crn_x_80e_coco.zip) | 640 640   |

* Convert to onnx

``` shell
python3 paddle_infer_shape.py  --model_dir ppyoloe_crn_s_300e_coco \
                             --model_filename model.pdmodel \
                             --params_filename model.pdiparams \
                             --save_dir new_model \
                             --input_shape_dict="{'image':[1,3,640,640], 'scale_factor':[1,2]}"

paddle2onnx  --model_dir new_model \
             --model_filename model.pdmodel \
             --params_filename model.pdiparams \
             --opset_version 13 \
             --save_file ppyoloe_crn_s_300e_coco.onnx

```

## Dataset

* [coco](http://images.cocodataset.org/zips/val2017.zip)

## References

* [PP-YOLOE: An evolved version of YOLO](https://arxiv.org/abs/2203.16250)
* [PaddlePaddle/ppyoloe](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/ppyoloe)
* [PaddleDetection](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/paddledetection)

## License

Apache 2.0
