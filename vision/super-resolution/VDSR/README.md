<!--- SPDX-License-Identifier:  -->

# VDSR

## Description

VDSR was proposed by Jiwon Kim et al. in 2016. The author mainly uses a deep convolutional network based on VGG-Net, which only learns residuals and use extremely high learning rates (104 times higher than SRCNN) enabled by adjustable gradient clipping, and ultimately has a great advantage in image quality performance.

## Model

|Model            |Download                                                   |PSNR (dB)          |
|-----------------|:----------------------------------------------------------|:------------------|
| VDSR            |[model](deploy.prototxt) [weight](_iter_10000.caffemodel)  |25.18~37.53        |

## Dataset

* [Accurate Image Super-Resolution Using Very Deep Convolutional Networks](https://cv.snu.ac.kr/research/VDSR/)

## References

* [Accurate Image Super-Resolution Using Very Deep Convolutional Networks](https://arxiv.org/abs/1511.04587)
* [BobLiu20/SuperResolution\_Caffe](https://github.com/BobLiu20/SuperResolution_Caffe)
* [Web](https://cv.snu.ac.kr/research/VDSR/)

## License

NO LICENSE
