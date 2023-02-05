<!--- SPDX-License-Identifier:  -->

# SRCNN

## Description

SRCNN is an Image SR reconstruction network proposed by the Chinese University of Hong Kong in 2014. The core structure is to use the CNN network to perform feature extraction and mapping on the original low-resolution image, and finally complete the high-resolution image reconstruction. Its essence is to use the deep learning neural network to achieve sparse autoencoder.

## Model

|Model            |Download                                                   |PSNR (dB)          |
|-----------------|:----------------------------------------------------------|:------------------|
| SRCNN           |[model](deploy.prototxt) [weight](srcnn.caffemodel)        |29.5~33            |

* Training: [Caffe code](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN/SRCNN_train.zip)
* Test: [Matlab code](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN/SRCNN_v1.zip)

## References

* [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092)
* [BobLiu20/SuperResolution_Caffe](https://github.com/BobLiu20/SuperResolution_Caffe)
* [Web](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)

## License

NO LICENSE
