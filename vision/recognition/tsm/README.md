<!--- SPDX-License-Identifier: Apache-2.0 -->

# TSM

## Description

This is a implementation of the Temporal Shift Module. This model enjoys both high efficiency and high performance on performing video understanding tasks.

## Model

|Model                      |Instance                                                      |
|---------------------------|--------------------------------------------------------------|
|[tsm.onnx](tsm.onnx)       |TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth|

We use TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth as an instance.

## Dataset

We sugguest use [tiny dataset](<https://github.com/Tramac/tiny-kinetics-400>) rather than large Kinetic-400.

## References

* ["TSM: Temporal Shift Module for Efficient Video Understanding"](<https://arxiv.org/abs/1811.08383>)

* [TSM](<https://github.com/mit-han-lab/temporal-shift-module>)

## License

Apache-2.0