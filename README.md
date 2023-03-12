# Cutoff - Cutting Off Prompt Effect

![cover](./images/cover.jpg)

## What is this?

This is an extension for [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) which limits the tokens' influence scope.

## Usage

1. Select `Enabled` checkbox.
2. Input words which you want to limit scope in `Target tokens`.
3. Generate images.

## Examples

```
7th_anime_v3_A-fp16 / DPM++ 2M Karras / 15 steps / 512x768
Prompt: a cute girl, white shirt with green tie, red shoes, blue hair, yellow eyes, pink skirt
Negative Prompt: (low quality, worst quality:1.4), nsfw
Target tokens: white, green, red, blue, yellow, pink
```

![sample 1](./images/sample-1.png)

![sample 2](./images/sample-2.png)

![sample 3](./images/sample-3.png)
