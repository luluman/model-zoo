<!--- SPDX-License-Identifier: Apache-2.0 -->

# stable_diffusion

## Description

Stable Diffusion is a latent text-to-image diffusion model.

## Model

Current model is openjourney model, which is based on stable diffusion 1.5.  
It contains 3 models: `clip text encoder`, `unet`, `vae decode`.

|Model                      |function                   |
|:--------------------------|:--------------------------|
|clip_text_encoder          |encoder the text           |
|unet                       |predict the noise          |
|vae_decode                 |decode the latent space    |
|stable_diffusion           |latent diffusion model     |

## References

* [open journey 1.4](https://huggingface.co/prompthero/openjourney-v4)

## License

Apache 2.0
