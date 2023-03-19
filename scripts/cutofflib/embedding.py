from dataclasses import dataclass
from itertools import product
import re
from typing import Union, List, Tuple
import numpy as np
import open_clip
from modules.sd_hijack_clip import FrozenCLIPEmbedderWithCustomWordsBase as CLIP
from modules import prompt_parser, shared
from scripts.cutofflib.utils import log

class ClipWrapper:
    def __init__(self, te: CLIP):
        self.te = te
        self.v1 = hasattr(te.wrapped, 'tokenizer')
        self.t = (
            te.wrapped.tokenizer if self.v1
            else open_clip.tokenizer._tokenizer
        )
    
    def token_to_id(self, token: str) -> int:
        if self.v1:
            return self.t._convert_token_to_id(token) # type: ignore
        else:
            return self.t.encoder[token]
    
    def id_to_token(self, id: int) -> str:
        if self.v1:
            return self.t.convert_ids_to_tokens(id) # type: ignore
        else:
            return self.t.decoder[id]
    
    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        if self.v1:
            return self.t.convert_ids_to_tokens(ids) # type: ignore
        else:
            return [self.t.decoder[id] for id in ids]
    
    def token(self, token: Union[int,str]):
        if isinstance(token, int):
            return Token(token, self.id_to_token(token))
        else:
            return Token(self.token_to_id(token), token)


@dataclass
class Token:
    id: int
    token: str

class CutoffPrompt:
    
    @staticmethod
    def _cutoff(prompt: str, clip: CLIP, tokens: List[str], padding: Token):
        pad = padding.token.replace('</w>', '')
        
        re_targets = [ re.compile(r'\b' + re.escape(x) + r'\b') for x in tokens ]
        replacer = [ ' ' + ' '.join([pad] * len(clip.tokenize(x))) + ' ' for x in tokens ]
        
        rows: List[Tuple[str,str]] = []
        for block in prompt.split(','):
            b0 = block
            for r, p in zip(re_targets, replacer):
                block = r.sub(p, block)
            b1 = block
            rows.append((b0, b1))
        
        return rows
    
    def __init__(self, prompt: str, clip: CLIP, tokens: List[str], padding: Token):
        self.prompt = prompt
        self.padding = padding
        rows = CutoffPrompt._cutoff(prompt, clip, tokens, padding)
        self.base = np.array([x[0] for x in rows])
        self.cut  = np.array([x[1] for x in rows])
        self.sw = np.array([False] * len(rows))
    
    @property
    def block_count(self):
        return self.base.shape[0]
    
    def switch(self, block_index: int, to: Union[bool,None] = None):
        if to is None:
            to = not self.sw[block_index]
        self.sw[block_index] = to
        return to
    
    def text(self, sw=None):
        if sw is None:
            sw = self.sw
        blocks = np.where(sw, self.cut, self.base)
        return ','.join(blocks)
    
    def active_blocks(self) -> np.ndarray:
        indices, = (self.base != self.cut).nonzero()
        return indices
    
    def generate(self):
        indices = self.active_blocks()
        for diff_sw in product([False, True], repeat=indices.shape[0]):
            sw = np.full_like(self.sw, False)
            sw[indices] = diff_sw
            yield diff_sw, self.text(sw)


def generate_prompts(
    clip: CLIP,
    prompt: str,
    targets: List[str],
    padding: Union[str,int,Token],
) -> CutoffPrompt:
    
    te = ClipWrapper(clip)
    
    if not isinstance(padding, Token):
        o_pad = padding
        padding = te.token(padding)
        if padding.id == clip.id_end:
            raise ValueError(f'`{o_pad}` is not a valid token.')
    
    result = CutoffPrompt(prompt, clip, targets, padding)
    
    log(f'[Cutoff] replace: {", ".join(targets)}')
    log(f'[Cutoff] to: {padding.token} ({padding.id})')
    log(f'[Cutoff] original: {prompt}')
    for i, (_, pp) in enumerate(result.generate()):
        log(f'[Cutoff]       #{i}: {pp}')
    
    return result


def token_to_block(clip: CLIP, prompt: str):
    te = ClipWrapper(clip)
    
    # cf. sd_hijack_clip.py
    
    parsed = prompt_parser.parse_prompt_attention(prompt)
    tokenized: List[List[int]] = clip.tokenize([text for text, _ in parsed])
    
    CHUNK_LENGTH = 75
    id_start = te.token(clip.id_start) # type: ignore
    id_end = te.token(clip.id_end) # type: ignore
    comma = te.token(',</w>')
    
    last_comma = -1
    current_block = 0
    current_tokens: List[Tuple[Token,int]] = []
    result: List[Tuple[Token,int]] = []
    
    def next_chunk():
        nonlocal current_tokens, last_comma
        
        to_add = CHUNK_LENGTH - len(current_tokens)
        if 0 < to_add:
            current_tokens += [(id_end, -1)] * to_add
        
        current_tokens = [(id_start, -1)] + current_tokens + [(id_end, -1)]
        
        last_comma = -1
        result.extend(current_tokens)
        current_tokens = []
        
    for tokens, (text, weight) in zip(tokenized, parsed):
        if text == 'BREAK' and weight == -1:
            next_chunk()
            continue
        
        p = 0
        while p < len(tokens):
            token = tokens[p]
            
            if token == comma.id:
                last_comma = len(current_tokens)
                current_block += 1
            
            elif (
                shared.opts.comma_padding_backtrack != 0 
                and len(current_tokens) == CHUNK_LENGTH
                and last_comma != -1 
                and len(current_tokens) - last_comma <= shared.opts.comma_padding_backtrack
            ):
                break_location = last_comma + 1
                reloc_tokens = current_tokens[break_location:]
                current_tokens = current_tokens[:break_location]
                next_chunk()
                current_tokens = reloc_tokens
            
            if len(current_tokens) == CHUNK_LENGTH:
                next_chunk()
            
            embedding, _ = clip.hijack.embedding_db.find_embedding_at_position(tokens, p)
            if embedding is None:
                if token == comma.id:
                    current_tokens.append((te.token(token), -1))
                else:
                    current_tokens.append((te.token(token), current_block))
                p += 1
                continue

            emb_len = int(embedding.vec.shape[0])
            if len(current_tokens) + emb_len > CHUNK_LENGTH:
                next_chunk()

            current_tokens += [(te.token(0), current_block)] * emb_len
            p += emb_len
            
    if len(current_tokens) > 0:
        next_chunk()
    
    return result
