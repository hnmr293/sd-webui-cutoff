from collections import defaultdict
from typing import Union, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import gradio as gr

from modules.processing import StableDiffusionProcessing
from modules import scripts

from scripts.cutofflib.sdhook import SDHook
from scripts.cutofflib.embedding import CLIP, generate_prompts, token_to_block
from scripts.cutofflib.utils import log, set_debug
from scripts.cutofflib.xyz import init_xyz

NAME = 'Cutoff'
PAD = '_</w>'

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    # cf. https://memo.sugyan.com/entry/2022/09/09/230645

    inputs_are_torch = False
    input_device = v0.device
    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


class Hook(SDHook):
    
    def __init__(
        self,
        enabled: bool,
        targets: List[str],
        padding: Union[str,int],
        weight: float,
        strong: bool,
        interpolate: str,
    ):
        super().__init__(enabled)
        self.targets = targets
        self.padding = padding
        self.weight = float(weight)
        self.strong = strong
        self.intp = interpolate
    
    def interpolate(self, t1: Tensor, t2: Tensor, w):
        if self.intp == 'lerp':
            return torch.lerp(t1, t2, w)
        else:
            return slerp(w, t1, t2)
    
    def hook_clip(self, p: StableDiffusionProcessing, clip: nn.Module):
        
        skip = False
        
        def hook(mod: nn.Module, inputs: Tuple[List[str]], output: Tensor):
            nonlocal skip
            
            if skip:
                return
            
            assert isinstance(mod, CLIP)
            
            prompts, *rest = inputs
            assert len(prompts) == output.shape[0]
            
            output = output.clone()
            for pidx, prompt in enumerate(prompts):
                tt = token_to_block(mod, prompt)
                
                cutoff = generate_prompts(mod, prompt, self.targets, self.padding)
                switch_base = np.full_like(cutoff.sw, self.strong)
                switch = np.full_like(cutoff.sw, True)
                active = cutoff.active_blocks()
                
                prompt_to_tokens = defaultdict(lambda: [])
                for tidx, (token, block_index) in enumerate(tt):
                    if block_index in active:
                        sw = switch.copy()
                        sw[block_index] = False
                        prompt = cutoff.text(sw)
                    else:
                        prompt = cutoff.text(switch_base)
                    prompt_to_tokens[prompt].append((tidx, token))
                
                #log(prompt_to_tokens)
                
                skip = True
                ks = list(prompt_to_tokens.keys())
                try:
                    vs = mod(ks)
                finally:
                    skip = False
                
                tensor = output[pidx, :, :] # e.g. (77, 768)
                for k, t in zip(ks, vs):
                    assert tensor.shape == t.shape
                    for tidx, token in prompt_to_tokens[k]:
                        log(f'{tidx:03} {token.token:<16} {k}')
                        tensor[tidx, :] = self.interpolate(tensor[tidx,:], t[tidx,:], self.weight)
                
            return output
        
        self.hook_layer(clip, hook)
    

class Script(scripts.Script):
    
    def __init__(self):
        super().__init__()
        self.last_hooker: Union[SDHook,None] = None

    def title(self):
        return NAME
    
    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        with gr.Accordion(NAME, open=False):
            enabled = gr.Checkbox(label='Enabled', value=False)
            targets = gr.Textbox(label='Target tokens (comma separated)', placeholder='red, blue')
            weight = gr.Slider(minimum=-1.0, maximum=2.0, step=0.01, value=0.5, label='Weight')
            with gr.Accordion('Details', open=False):
                strong = gr.Checkbox(value=False, label='Cutoff strongly.')
                padding = gr.Textbox(label='Padding token (ID or single token)')
                lerp = gr.Radio(choices=['Lerp', 'SLerp'], value='Lerp', label='Interpolation method')
            
            debug = gr.Checkbox(value=False, label='Debug log')
            debug.change(fn=set_debug, inputs=[debug], outputs=[])
                
        return [
            enabled,
            targets,
            weight,
            strong,
            padding,
            lerp,
            debug,
        ]
    
    def process(
        self,
        p: StableDiffusionProcessing,
        enabled: bool,
        targets_: str,
        weight: Union[float,int],
        strong: bool,
        padding: Union[str,int],
        intp: str,
        debug: bool,
    ):
        set_debug(debug)
        
        if self.last_hooker is not None:
            self.last_hooker.__exit__(None, None, None)
            self.last_hooker = None
        
        if not enabled:
            return
        
        if targets_ is None or len(targets_) == 0:
            return
        
        targets = [x.strip() for x in targets_.split(',')]
        
        if padding is None:
            padding = PAD
        elif isinstance(padding, str):
            if len(padding) == 0:
                padding = PAD
            else:
                try:
                    padding = int(padding)
                except:
                    if not padding.endswith('</w>'):
                        padding += '</w>'
        
        weight = float(weight)
        intp = intp.lower()
        
        self.last_hooker = Hook(
            enabled=True,
            targets=targets,
            padding=padding,
            weight=weight,
            strong=strong,
            interpolate=intp,
        )
        
        self.last_hooker.setup(p)
        self.last_hooker.__enter__()
        
        p.extra_generation_params.update({
            f'{NAME} Enabled': enabled,
            f'{NAME} targets': targets,
            f'{NAME} padding': padding,
            f'{NAME} weight': weight,
            f'{NAME} strong': strong,
        })

init_xyz(Script, NAME)
