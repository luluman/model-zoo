<!--- SPDX-License-Identifier: Apache-2.0 -->

# CLIP

## Description
CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs.
It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3.

## Model

|Model            |Download                                       |
|-----------------|-----------------------------------------------|
|CLIP             | [336M](clip.onnx)                             |

## Dataset

[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html).

## References

* **CLIP** model was described in the paper titled [Learning Transferable Visual Models From Natural Language Supervision](<https://arxiv.org/abs/2103.00020>)
* The pre-train model is from the Project: (<https://github.com/openai/CLIP>).

## License

Apache 2.0: