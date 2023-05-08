<!--- SPDX-License-Identifier: Apache-2.0 -->

# SegFormer

## Description

SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.
[website](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/segformer)

## Model

|Model                |Download                                                                    |
|---------------------|:---------------------------------------------------------------------------|
|SegFormer            |[14.4 MB](segformer.onnx)                                                   |

* Convert to onnx in mmsegmentation

``` shell
python tools/pytorch2onnx.py 
          local_configs/segformer/B0/segformer.b0.512x512.ade.160k.py
          --checkpoint onnx/segformer.b0.512x512.ade.160k.pth
          --show
          --verify
          --output-file onnx/segformer.onnx
          --opset-version 12
          --shape 512 512

```

## Dataset

[ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

## References

* [SegFormer](https://github.com/NVlabs/SegFormer)

## License

Apache-2.0
