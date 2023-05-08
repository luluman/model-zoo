<!--- SPDX-License-Identifier: Apache License 2.0 -->

# Vit-large-patch16-384

## Description

The Vision Transformer (ViT) is a transformer encoder model (BERT-like) pretrained on a large collection of images in a supervised fashion, namely ImageNet-21k, at a resolution of 224x224 pixels. Next, the model was fine-tuned on ImageNet (also referred to as ILSVRC2012), a dataset comprising 1 million images and 1,000 classes, at a higher resolution of 384x384. For more information see [google/vit-large-patch16-384](https://huggingface.co/google/vit-large-patch16-384).

## Model

|Model                     |Download                   |
|--------------------------|:--------------------------|
|vit-large-patch16-384     |[1.13 GB](vit_L_16.onnx)   |

## Dataset

[ImageNet-21k](http://www.image-net.org/)

[ImageNet](http://www.image-net.org/challenges/LSVRC/2012/)

## References

* Vision Transformer (ViT) model was introduced in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

* Vision Transformer (ViT) model first released in [this repository](https://github.com/google-research/vision_transformer). Howerver, the weights were converted from the [timm repository](https://github.com/rwightman/pytorch-image-models)

## License

Apache License 2.0
