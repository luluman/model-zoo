<!--- SPDX-License-Identifier: MIT -->

# ERFNet

## Description

ERFnet is a semantic segmentation network based on residual connections and separable convolutions designed to improve the efficiency of the model in processing frames without reducing accuracy

## Model

|Model       |Download                                                                                |Class IoU (%)      |Category IoU (%)   |
|------------|:---------------------------------------------------------------------------------------|:------------------|:------------------|
|ERFNet      |[model](erfnet_deploy_mergebn.prototxt) [weight](erfnet_cityscapes_mergebn.caffemodel)  | 69.7              | 87.3              |

## Dataset

cityscapes dataset: [website address](https://www.cityscapes-dataset.com/) [git address](https://github.com/mcordts/cityscapesScripts)

## References

* [ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation](https://ieeexplore.ieee.org/document/8063438/)
* [Yuelong-Yu/ERFNet-Caffe](https://github.com/Yuelong-Yu/ERFNet-Caffe)

## License

MIT
