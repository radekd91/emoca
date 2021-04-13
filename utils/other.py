import sys


def class_from_str(str, module=None) -> type:
    if module is None:
        module = sys.modules[__name__]
    cl = getattr(module, str)
    return cl