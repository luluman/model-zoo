<!--- SPDX-License-Identifier: Apache 2.0 -->
<!-- markdownlint-disable MD013 -->

# BiSeNet

## Description

BiSeNetv2 is a state-of-the-art deep learning network model for semantic segmentation tasks. It was developed by researchers at the University of Chinese Academy of Sciences and Megvii Research.

The BiSeNetv2 model is designed to achieve high accuracy and efficiency in real-time semantic segmentation tasks. It uses a two-pathway architecture, consisting of a spatial pathway and a context pathway, to extract features from different scales and resolutions of the input image. The spatial pathway is used to capture local details and fine-grained features, while the context pathway is used to capture global contextual information.

The BiSeNetv2 model also uses a multi-scale fusion strategy to combine features from different scales and resolutions, which improves the accuracy and robustness of the segmentation results. Additionally, it incorporates a lightweight module called the Spatial Path Enhancer (SPE), which helps to enhance the spatial pathway's ability to capture fine details and improve the overall segmentation performance.

Overall, BiSeNetv2 is a powerful and efficient network model that has achieved state-of-the-art performance on several benchmark datasets for semantic segmentation, including Cityscapes, COCO-Stuff, and ADE20K.

## Model

| Model           | Download                                       | Shape(hw) |
| --------------- |:---------------------------------------------- |:--------- |
| mobileseg       | [13.5MB](bisenetv2_city.onnx)                  | 1024 2048 |

## Dataset

* [cityscapes](https://www.cityscapes-dataset.com/)

## References

* [BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation](https://arxiv.org/abs/2004.02147)
* [CoinCheung/BiSeNet](https://github.com/CoinCheung/BiSeNet)

## License

Apache 2.0

<!-- markdownlint-enable MD013 -->