import sys
from typing import Any, Callable, Union

from torch import nn
from torch.utils.hooks import RemovableHandle

from ldm.modules.diffusionmodules.openaimodel import (
    TimestepEmbedSequential,
)
from ldm.modules.attention import (
    SpatialTransformer,
    BasicTransformerBlock,
    CrossAttention,
    MemoryEfficientCrossAttention,
)
from ldm.modules.diffusionmodules.openaimodel import (
    ResBlock,
)
from modules.processing import StableDiffusionProcessing
from modules import shared

class ForwardHook:
    
    def __init__(self, module: nn.Module, fn: Callable[[nn.Module, Callable[..., Any], Any], Any]):
        self.o = module.forward
        self.fn = fn
        self.module = module
        self.module.forward = self.forward
    
    def remove(self):
        if self.module is not None and self.o is not None:
            self.module.forward = self.o
            self.module = None
            self.o = None
        self.fn = None
    
    def forward(self, *args, **kwargs):
        if self.module is not None and self.o is not None:
            if self.fn is not None:
                return self.fn(self.module, self.o, *args, **kwargs)
        return None
        

class SDHook:
    
    def __init__(self, enabled: bool):
        self._enabled = enabled
        self._handles: list[Union[RemovableHandle,ForwardHook]] = []
    
    @property
    def enabled(self):
        return self._enabled
    
    @property
    def batch_num(self):
        return shared.state.job_no
    
    @property
    def step_num(self):
        return shared.state.current_image_sampling_step
    
    def __enter__(self):
        if self.enabled:
            pass
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self.dispose()
    
    def dispose(self):
        pass
    
    def setup(
        self,
        p: StableDiffusionProcessing
    ):
        if not self.enabled:
            return
        
        wrapper = getattr(p.sd_model, "model", None)
        
        unet: Union[nn.Module,None] = getattr(wrapper, "diffusion_model", None) if wrapper is not None else None
        vae: Union[nn.Module,None] = getattr(p.sd_model, "first_stage_model", None)
        clip: Union[nn.Module,None] = getattr(p.sd_model, "cond_stage_model", None)
        
        assert unet is not None, "p.sd_model.diffusion_model is not found. broken model???"
        self._do_hook(p, p.sd_model, unet=unet, vae=vae, clip=clip) # type: ignore
        self.on_setup()
    
    def on_setup(self):
        pass
    
    def _do_hook(
        self,
        p: StableDiffusionProcessing,
        model: Any,
        unet: Union[nn.Module,None],
        vae: Union[nn.Module,None],
        clip: Union[nn.Module,None]
    ):
        assert model is not None, "empty model???"
        
        if clip is not None:
            self.hook_clip(p, clip)
        
        if unet is not None:
            self.hook_unet(p, unet)
        
        if vae is not None:
            self.hook_vae(p, vae)
    
    def hook_vae(
        self,
        p: StableDiffusionProcessing,
        vae: nn.Module
    ):
        pass

    def hook_unet(
        self,
        p: StableDiffusionProcessing,
        unet: nn.Module
    ):
        pass

    def hook_clip(
        self,
        p: StableDiffusionProcessing,
        clip: nn.Module
    ):
        pass

    def hook_layer(
        self,
        module: Union[nn.Module,Any],
        fn: Callable[[nn.Module, tuple, Any], Any]
    ):
        if not self.enabled:
            return
        
        assert module is not None
        assert isinstance(module, nn.Module)
        self._handles.append(module.register_forward_hook(fn))

    def hook_layer_pre(
        self,
        module: Union[nn.Module,Any],
        fn: Callable[[nn.Module, tuple], Any]
    ):
        if not self.enabled:
            return
        
        assert module is not None
        assert isinstance(module, nn.Module)
        self._handles.append(module.register_forward_pre_hook(fn))

    def hook_forward(
        self,
        module: Union[nn.Module,Any],
        fn: Callable[[nn.Module, Callable[..., Any], Any], Any]
    ):
        assert module is not None
        assert isinstance(module, nn.Module)
        self._handles.append(ForwardHook(module, fn))
    
    def log(self, msg: str):
        print(msg, file=sys.stderr)


# enumerate SpatialTransformer in TimestepEmbedSequential 
def each_transformer(unet_block: TimestepEmbedSequential):
    for block in unet_block.children():
        if isinstance(block, SpatialTransformer):
            yield block

# enumerate BasicTransformerBlock in SpatialTransformer
def each_basic_block(trans: SpatialTransformer):
    for block in trans.transformer_blocks.children():
        if isinstance(block, BasicTransformerBlock):
            yield block

# enumerate Attention Layers in TimestepEmbedSequential
# each_transformer + each_basic_block
def each_attns(unet_block: TimestepEmbedSequential):
    for n, trans in enumerate(each_transformer(unet_block)):
        for depth, basic_block in enumerate(each_basic_block(trans)):
            # attn1: Union[CrossAttention,MemoryEfficientCrossAttention]
            # attn2: Union[CrossAttention,MemoryEfficientCrossAttention]
            
            attn1, attn2 = basic_block.attn1, basic_block.attn2
            assert isinstance(attn1, CrossAttention) or isinstance(attn1, MemoryEfficientCrossAttention)
            assert isinstance(attn2, CrossAttention) or isinstance(attn2, MemoryEfficientCrossAttention)
            
            yield n, depth, attn1, attn2

def each_unet_attn_layers(unet: nn.Module):
    def get_attns(layer_index: int, block: TimestepEmbedSequential, format: str):
        for n, d, attn1, attn2 in each_attns(block):
            kwargs = {
                'layer_index': layer_index,
                'trans_index': n,
                'block_index': d
            }
            yield format.format(attn_name='sattn', **kwargs), attn1
            yield format.format(attn_name='xattn', **kwargs), attn2
    
    def enumerate_all(blocks: nn.ModuleList, format: str):
        for idx, block in enumerate(blocks.children()):
            if isinstance(block, TimestepEmbedSequential):
                yield from get_attns(idx, block, format)
    
    inputs: nn.ModuleList = unet.input_blocks           # type: ignore
    middle: TimestepEmbedSequential = unet.middle_block # type: ignore
    outputs: nn.ModuleList = unet.output_blocks         # type: ignore
    
    yield from enumerate_all(inputs, 'IN{layer_index:02}_{trans_index:02}_{block_index:02}_{attn_name}')
    yield from get_attns(0, middle, 'M{layer_index:02}_{trans_index:02}_{block_index:02}_{attn_name}')
    yield from enumerate_all(outputs, 'OUT{layer_index:02}_{trans_index:02}_{block_index:02}_{attn_name}')


def each_unet_transformers(unet: nn.Module):
    def get_trans(layer_index: int, block: TimestepEmbedSequential, format: str):
        for n, trans in enumerate(each_transformer(block)):
            kwargs = {
                'layer_index': layer_index,
                'block_index': n,
                'block_name': 'trans',
            }
            yield format.format(**kwargs), trans
    
    def enumerate_all(blocks: nn.ModuleList, format: str):
        for idx, block in enumerate(blocks.children()):
            if isinstance(block, TimestepEmbedSequential):
                yield from get_trans(idx, block, format)
    
    inputs: nn.ModuleList = unet.input_blocks           # type: ignore
    middle: TimestepEmbedSequential = unet.middle_block # type: ignore
    outputs: nn.ModuleList = unet.output_blocks         # type: ignore
    
    yield from enumerate_all(inputs, 'IN{layer_index:02}_{block_index:02}_{block_name}')
    yield from get_trans(0, middle, 'M{layer_index:02}_{block_index:02}_{block_name}')
    yield from enumerate_all(outputs, 'OUT{layer_index:02}_{block_index:02}_{block_name}')


def each_resblock(unet_block: TimestepEmbedSequential):
    for block in unet_block.children():
        if isinstance(block, ResBlock):
            yield block

def each_unet_resblock(unet: nn.Module):
    def get_resblock(layer_index: int, block: TimestepEmbedSequential, format: str):
        for n, res in enumerate(each_resblock(block)):
            kwargs = {
                'layer_index': layer_index,
                'block_index': n,
                'block_name': 'resblock',
            }
            yield format.format(**kwargs), res
    
    def enumerate_all(blocks: nn.ModuleList, format: str):
        for idx, block in enumerate(blocks.children()):
            if isinstance(block, TimestepEmbedSequential):
                yield from get_resblock(idx, block, format)
    
    inputs: nn.ModuleList = unet.input_blocks           # type: ignore
    middle: TimestepEmbedSequential = unet.middle_block # type: ignore
    outputs: nn.ModuleList = unet.output_blocks         # type: ignore
    
    yield from enumerate_all(inputs, 'IN{layer_index:02}_{block_index:02}_{block_name}')
    yield from get_resblock(0, middle, 'M{layer_index:02}_{block_index:02}_{block_name}')
    yield from enumerate_all(outputs, 'OUT{layer_index:02}_{block_index:02}_{block_name}')

