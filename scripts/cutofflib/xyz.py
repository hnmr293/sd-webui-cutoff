import os
from typing import Union, List, Callable

from modules import scripts
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img


def __set_value(p: StableDiffusionProcessing, script: type, index: int, value):
    args = list(p.script_args)
    
    if isinstance(p, StableDiffusionProcessingTxt2Img):
        all_scripts = scripts.scripts_txt2img.scripts
    else:
        all_scripts = scripts.scripts_img2img.scripts
    
    froms = [x.args_from for x in all_scripts if isinstance(x, script)]
    for idx in froms:
        assert idx is not None
        args[idx + index] = value
    
    p.script_args = type(p.script_args)(args)


def to_bool(v: str):
    if len(v) == 0: return False
    v = v.lower()
    if 'true' in v: return True
    if 'false' in v: return False
    
    try:
        w = int(v)
        return bool(w)
    except:
        acceptable = ['True', 'False', '1', '0']
        s = ', '.join([f'`{v}`' for v in acceptable])
        raise ValueError(f'value must be one of {s}.')


class AxisOptions:
    
    def __init__(self, AxisOption: type, axis_options: list):
        self.AxisOption = AxisOption
        self.target = axis_options
        self.options = []
    
    def __enter__(self):
        self.options.clear()
        return self
    
    def __exit__(self, ex_type, ex_value, trace):
        if ex_type is not None:
            return
        
        for opt in self.options:
            self.target.append(opt)
        
        self.options.clear()
    
    def create(self, name: str, type_fn: Callable, action: Callable, choices: Union[List[str],None]):
        if choices is None or len(choices) == 0:
            opt = self.AxisOption(name, type_fn, action)
        else:
            opt = self.AxisOption(name, type_fn, action, choices=lambda: choices)
        return opt
    
    def add(self, axis_option):
        self.target.append(axis_option)


__init = False

def init_xyz(script: type, ext_name: str):
    global __init
    
    if __init:
        return
    
    for data in scripts.scripts_data:
        name = os.path.basename(data.path)
        if name != 'xy_grid.py' and name != 'xyz_grid.py':
            continue
        
        if not hasattr(data.module, 'AxisOption'):
            continue
        
        if not hasattr(data.module, 'axis_options'):
            continue
        
        AxisOption = data.module.AxisOption
        axis_options = data.module.axis_options
        
        if not isinstance(AxisOption, type):
            continue
        
        if not isinstance(axis_options, list):
            continue
        
        try:
            create_options(ext_name, script, AxisOption, axis_options)
        except:
            pass
            
    __init = True


def create_options(ext_name: str, script: type, AxisOptionClass: type, axis_options: list):
    with AxisOptions(AxisOptionClass, axis_options) as opts:
        def define(param: str, index: int, type_fn: Callable, choices: List[str] = []):
            def fn(p, x, xs):
                __set_value(p, script, index, x)
            
            name = f'[{ext_name}] {param}'
            return opts.create(name, type_fn, fn, choices)
        
        options = [
            define('Enabled', 0, to_bool, choices=['false', 'true']),
            define('Targets', 1, str),
            define('Weight', 2, float),
            define('Disable for Negative Prompt', 3, to_bool, choices=['false', 'true']),
            define('Strong', 4, to_bool, choices=['false', 'true']),
            define('Padding', 5, str),
            define('Interpolation', 6, str, choices=['Lerp', 'SLerp']),
        ]
        
        for opt in options:
            opts.add(opt)
