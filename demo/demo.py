'''
Author: Juncfang
Date: 2023-05-09 10:21:53
LastEditTime: 2023-05-09 14:03:36
LastEditors: Juncfang
Description: 
FilePath: /sd-webui-cutoff/demo/demo.py
 
'''
import os
import sys
import torch
import random
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.cutoff_diffusers import cutoff_text_encoder

# init
random.seed(0)
model_dir = "./anything-v4.0" # turnoff safety_checker by modify the model_index.json
pipe = StableDiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.float16)
pipe.to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(model_dir, subfolder="scheduler")
seed_list = [random.randint(0, 666654) for _ in range(4)]

# parameters
prompt = "a cute girl, white shirt with green tie, black shoes, red hair, yellow eyes, green skirt"
negative_prompt = "nsfw, low quality, worst quality, nsfw"
guidance_scale = 7
width = 512
height = 768
num_inference_steps = 30

# for original inference
for seed in seed_list:
    image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device='cuda').manual_seed(seed),
            ).images[0]
    image.save(f"../images/org_{seed}.png")

# for cutoff inference
text_encoder = CLIPTextModel.from_pretrained(
        model_dir, 
        subfolder="text_encoder", 
    )
text_encoder.to("cuda")
tokenizer = CLIPTokenizer.from_pretrained(
    model_dir, 
    subfolder="tokenizer",
)

for seed in seed_list:
    tensor = cutoff_text_encoder(
        [prompt], 
        text_encoder, 
        tokenizer, 
        targets=['red', 'blue', 'white', 'green', 'yellow', 'pink', 'black', 'gray', 'orange', 'purple', 'cyan', 'brown']
        )
    tk = tokenizer(
        negative_prompt, 
        max_length=tokenizer.model_max_length, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt").to(text_encoder.device)
    tensor_neg = text_encoder(tk["input_ids"])[0]
    image = pipe(
        prompt_embeds=tensor,
        negative_prompt_embeds=tensor_neg,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device='cuda').manual_seed(seed),
    ).images[0]
    image.save(f"../images/cutoff_{seed}.png")