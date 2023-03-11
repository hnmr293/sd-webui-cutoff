import sys

_debug = False

def set_debug(is_debug: bool):
    global _debug
    _debug = is_debug

def log(s: str):
    if _debug:
        print(s, file=sys.stderr)
