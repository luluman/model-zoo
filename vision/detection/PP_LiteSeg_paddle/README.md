<!--- SPDX-License-Identifier: Apache-2.0 -->

# PP-LiteSeg

## Description

We propose PP-LiteSeg, a novel lightweight model for the real-time semantic segmentation task. Specifically, we present a Flexible and Lightweight Decoder (FLD) to reduce computation overhead of previous decoder.  
To strengthen feature representations, we propose a Unified Attention Fusion Module (UAFM), which takes advantage of spatial and channel attention to produce a weight and then fuses the input features with the weight. Moreover, a Simple Pyramid Pooling Module (SPPM) is proposed to aggregate global context with low computation cost.  

Paddle2ONNX needs to be installed for the conversion of model
```shell
#1. change the Paddle model from dynamic inputs to fixed inputs, see paddle_infer_shape.py
python paddle_infer_shape.py --model_dir PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer \
                             --model_filename model.pdmodel \
                             --params_filename model.pdiparams \
                             --save_dir pp_liteseg_fix \
                             --input_shape_dict="{'x':[1,3,512,512]}"

#2. Convert the Paddle model with fixed inputs to the ONNX model, see paddle2onnx
paddle2onnx --model_dir pp_liteseg_fix \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file pp_liteseg.onnx \
            --enable_dev_version True
```

## Model

|Model                                            |Download    |mIoU (%)       |mIoU (flip) (%)|mIoU (ms+flip)(%)|
|-------------------------------------------------|:-----------|:--------------|:--------------|:----------------|
| PP-LiteSeg-B(STDC2)-cityscapes-without-argmax   |31 MB       |79.04%         |79.52%         |79.85%           |

## Dataset

[Cityscapes](https://paddleseg.bj.bcebos.com/dataset/cityscapes.tar), [CamVid](https://paddleseg.bj.bcebos.com/dataset/camvid.tar).

## References

* [PP-LiteSeg: A Superior Real-Time Semantic Segmentation Model](https://arxiv.org/abs/2204.02681)  
* [PaddlePaddle/PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/configs/pp_liteseg/README.md)
* [PP-LiteSeg-B(STDC2)-cityscapes-without-argmax](https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz)

## License

Apache 2.0
