<!--- SPDX-License-Identifier: Apache-2.0 -->

# PaddleSeg

## Description

PaddleSeg is an end-to-end high-efficent development toolkit for image
segmentation based on PaddlePaddle, which helps both developers and researchers
in the whole process of designing segmentation models, training models,
optimizing performance and inference speed, and deploying models. A lot of
well-trained models and various real-world applications in both industry and
academia help users conveniently build hands-on experiences in image
segmentation.
[website](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/segmentation/paddleseg)

It is recommended to use fastdeploy.

## Model

|Model                |Download                              |Shape(hw)     |mIoU      |
|---------------------|:-------------------------------------|:-------------|:---------|
|PaddleSeg            |[29 MB](pp_humansegv2_mobile.onnx)    |192 192       |93.13%    |

* Convert to onnx

``` shell
python3 paddle_infer_shape.py  --model_dir . \
                             --model_filename model.pdmodel \
                             --params_filename model.pdiparams \
                             --save_dir new_model \
                             --input_shape_dict="{'x':[1,3,192,192]}"

paddle2onnx  --model_dir new_model \
             --model_filename model.pdmodel \
             --params_filename model.pdiparams \
             --opset_version 13 \
             --save_file pp_humansegv2_mobile.onnx

```

## Dataset

[PP-HumanSeg14K](https://github.com/PaddlePaddle/PaddleSeg/blob/19351bab9a824a8f96e1c1b527ec2d7db21309c9/contrib/PP-HumanSeg/paper.md#pp-humanseg14k-a-large-scale-teleconferencing-video-dataset)
[matting_human_datasets](https://github.com/aisegmentcn/matting_human_datasets)

## References

* [PP-HumanSegV2-Mobile-without-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/PP_HumanSegV2_Mobile_192x192_infer.tgz)

## License

Apache-2.0
