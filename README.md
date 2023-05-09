<!--
 * @Author: Juncfang
 * @Date: 2023-05-09 10:12:38
 * @LastEditTime: 2023-05-09 14:28:02
 * @LastEditors: Juncfang
 * @Description: 
 * @FilePath: /sd-webui-cutoff/README.md
 *  
-->
# Cutoff - Cutting Off Prompt Effect

![images](./images/compare.jpg)

## What is this?

This is an cutoff extension for [diffusers](https://github.com/huggingface/diffusers) . This code is simply reorganize the sd-webui-cutoff to diffusers way.

## Usage

Just look at ![demo.py](./demo/demo.py)

## Note
I download `anything-v4.0` at demo/anything-v4.0 and turnoff safety_checker by modify the model_index.json (delete the line about safety_checker). You can found model_index.json at huggingface model cache and modify it if you wanna reproduce the experiment. You can also save anything-v4.0 to some path like this:
```python

from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained('anything-v4.0')
pipe.save_pretrained("/path/to/save")
