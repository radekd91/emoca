import sys
from pathlib import Path


def class_from_str(str, module=None, none_on_fail = False) -> type:
    if module is None:
        module = sys.modules[__name__]
    if hasattr(module, str):
        cl = getattr(module, str)
        return cl
    elif str.lower() == 'none' or none_on_fail:
        return None
    raise RuntimeError(f"Class '{str}' not found.")


def get_path_to_assets() -> Path:
    import gdl
    return Path(gdl.__file__).parents[1] / "assets"


def get_path_to_externals() -> Path:
    import gdl
    return Path(gdl.__file__).parents[1] / "external"
