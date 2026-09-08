__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


def __getattr__(name):
    """Keep engine adapters importable without loading the ComfyUI node catalog."""
    if name in __all__:
        from . import nodes

        return getattr(nodes, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
