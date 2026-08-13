"""Lazy public exports for the optional LTX 2.5 MLX runtime."""

from importlib import import_module

_EXPORT_MODULES = {
    "LTX25_CONFIG_MODES": ".runtime",
    "LTX25_GENERATION_PRESETS": ".runtime",
    "LTX25ComponentSpec": ".runtime",
    "LTX25GenerationConfig": ".runtime",
    "LTX25RuntimeCache": ".runtime",
    "apply_ltx25_generation_preset": ".runtime",
    "backend_capability": ".runtime",
    "LTX25Model": ".transformer",
    "LTX25TransformerConfig": ".transformer",
    "load_ltx25_transformer": ".transformer",
    "transformer_metadata": ".transformer",
    "LTX25DenoiseOutput": ".sampling",
    "euler_ancestral_denoise_loop": ".sampling",
    "euler_ancestral_step": ".sampling",
    "inspect_gemma4_pack": ".gemma_pack",
    "gemma4_mlx_model_config": ".gemma_pack",
    "remap_gemma4_weight_key": ".gemma_pack",
    "load_gemma4_backbone": ".gemma_encoder",
    "collect_gemma4_hidden_states": ".gemma_encoder",
    "load_gemma4_feature_extractor": ".gemma_encoder",
    "load_gemma4_tokenizer": ".gemma_encoder",
    "resolve_prompt_context_length": ".gemma_encoder",
    "tokenize_gemma4": ".gemma_encoder",
    "LTX25Gemma4Conditioner": ".gemma_encoder",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORT_MODULES})


__all__ = list(_EXPORT_MODULES)
