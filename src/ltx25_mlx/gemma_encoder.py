"""Direct MLX construction and loading for the LTX 2.5 Gemma 4 backbone."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Any

from safetensors import safe_open

from .gemma_pack import (
    GEMMA_CONFIG_METADATA_KEY,
    gemma4_mlx_model_config,
    inspect_gemma4_pack,
    remap_gemma4_weight_key,
)
from .transformer import precompute_rope_freqs_float64


def _ltx25_connector_class():
    from ltx_core_mlx.text_encoders.gemma.embeddings_connector import (
        Embeddings1DConnector,
        _rms_norm,
    )

    class LTX25Embeddings1DConnector(Embeddings1DConnector):
        def __call__(self, hidden_states, attention_mask=None):
            import mlx.core as mx
            from ltx_core_mlx.utils.memory import aggressive_cleanup

            batch = hidden_states.shape[0]
            if self.num_registers > 0:
                repeats = math.ceil(
                    max(1024, hidden_states.shape[1]) / self.num_registers
                )
                registers = mx.tile(self.learnable_registers, (repeats, 1))
                register_tail = registers[hidden_states.shape[1] :]
                if register_tail.shape[0]:
                    register_tail = mx.broadcast_to(
                        register_tail[None, :, :],
                        (batch, register_tail.shape[0], self.dim),
                    )
                    hidden_states = mx.concatenate([hidden_states, register_tail], axis=1)
                # Official LTX 2.5 makes every real prompt and learned register
                # token visible to the connector. The Gemma mask is intentionally
                # not forwarded after the register bank is appended.
            positions = mx.arange(hidden_states.shape[1]).astype(mx.float32)[None, :, None]
            rope_freqs = precompute_rope_freqs_float64(
                positions,
                inner_dim=self.dim,
                num_heads=self.dim // self.head_dim,
                theta=10000.0,
                max_pos=[self.max_pos],
                rope_type="split",
            )
            for block in self.transformer_1d_blocks:
                hidden_states = block(hidden_states, rope_freqs=rope_freqs)
                mx.eval(hidden_states)
                aggressive_cleanup()
            return _rms_norm(hidden_states)

    return LTX25Embeddings1DConnector


def load_gemma4_backbone(path: str | Path, *, _pack_weights=None):
    """Construct and strictly load the unified Gemma 4 language backbone.

    Tensor loading remains lazy under MLX. Non-language multimodal towers,
    embedded tokenizer bytes, and LTX connector weights are excluded before
    the model is populated. The caller owns the returned model lifecycle.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models.gemma4 import Model, ModelArgs

    source = Path(path).expanduser()
    report = inspect_gemma4_pack(source)
    paged_manifest = None
    if source.is_dir():
        from .paged_checkpoint import LTX25PagedManifest

        paged_manifest = LTX25PagedManifest.load(source)
        metadata = paged_manifest.metadata
    else:
        with safe_open(source, framework="numpy") as handle:
            metadata = handle.metadata() or {}
    raw_config = metadata.get(GEMMA_CONFIG_METADATA_KEY)
    if raw_config is None:  # inspect_gemma4_pack produces the detailed user-facing error.
        raise ValueError(f"LTX 2.5 Gemma pack {source} is missing Gemma configuration.")
    gemma_config: dict[str, Any] = (
        raw_config if isinstance(raw_config, dict) else json.loads(raw_config)
    )
    mlx_config = gemma4_mlx_model_config(gemma_config)
    model = Model(ModelArgs.from_dict(mlx_config))

    if paged_manifest is not None:
        fixed_weights = dict(mx.load(str(paged_manifest.fixed_path)))
        mapped = {}
        for key, value in fixed_weights.items():
            mapped_key = remap_gemma4_weight_key(key, layout=str(report["weight_layout"]))
            if mapped_key is not None:
                mapped[mapped_key] = value
        inner = model.language_model.model
        previous_kvs = list(inner.previous_kvs)
        expected = {
            key for key, _value in tree_flatten(model.parameters()) if ".layers." not in key
        }
        missing = sorted(expected - mapped.keys())
        unexpected = sorted(mapped.keys() - expected)
        if missing or unexpected:
            raise ValueError(
                "Paged Gemma 4 fixed weight mismatch: "
                f"missing={missing[:8]}; unexpected={unexpected[:8]}"
            )
        model.load_weights(list(mapped.items()), strict=False)
        model.eval()
        object.__setattr__(model, "_weetodd_paged_manifest", paged_manifest)
        object.__setattr__(model, "_weetodd_previous_kvs", previous_kvs)
        object.__setattr__(model, "_weetodd_weight_layout", str(report["weight_layout"]))
        from .page_prefetch import LTX25PagePrefetch

        object.__setattr__(
            model,
            "_weetodd_page_prefetch",
            LTX25PagePrefetch(
                paged_manifest.root,
                paged_manifest.layers,
                enabled=LTX25PagePrefetch.default_enabled(),
                thread_name="ltx25-gemma-prefetch",
            ),
        )
        return model, mlx_config, report

    pack_weights = _pack_weights if _pack_weights is not None else mx.load(str(source))
    mapped = {}
    for key, value in pack_weights.items():
        mapped_key = remap_gemma4_weight_key(key, layout=str(report["weight_layout"]))
        if mapped_key is not None:
            mapped[mapped_key] = value

    expected = {key for key, _value in tree_flatten(model.parameters())}
    actual = set(mapped)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing[:8]))
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected[:8]))
        raise ValueError("Gemma 4 MLX backbone weight mismatch: " + "; ".join(details))

    model.load_weights(list(mapped.items()), strict=True)
    model.eval()
    return model, mlx_config, report


def collect_gemma4_hidden_states(
    model,
    token_ids,
    attention_mask=None,
    *,
    progress_callback=None,
    is_cancelled=None,
):
    """Collect embedding plus per-layer Gemma 4 states with bounded MLX graphs."""
    import mlx.core as mx
    from mlx_lm.models.base import create_causal_mask

    inner = model.language_model.model
    if attention_mask is None:
        attention_mask = mx.ones(token_ids.shape, dtype=mx.int32)
    if tuple(attention_mask.shape) != tuple(token_ids.shape):
        raise ValueError("Gemma 4 attention_mask must match token_ids shape.")

    hidden = inner.embed_tokens(token_ids) * inner.embed_scale
    states = [hidden]
    left_padding = token_ids.shape[1] - attention_mask.sum(axis=-1)
    mask_by_type = {
        "full_attention": create_causal_mask(token_ids.shape[1], left_padding=left_padding),
        "sliding_attention": create_causal_mask(
            token_ids.shape[1],
            window_size=inner.window_size,
            left_padding=left_padding,
        ),
    }
    paged_manifest = getattr(model, "_weetodd_paged_manifest", None)
    total = paged_manifest.num_layers if paged_manifest is not None else len(inner.layers)
    previous_kvs = getattr(model, "_weetodd_previous_kvs", inner.previous_kvs)
    intermediates = [(None, None)] * total
    page_prefetch = getattr(model, "_weetodd_page_prefetch", None)
    if page_prefetch is not None:
        page_prefetch.start(0)
    for index in range(total):
        previous_index = previous_kvs[index]
        if is_cancelled is not None and is_cancelled():
            raise InterruptedError("LTX 2.5 Gemma 4 encoding cancelled.")
        if paged_manifest is None:
            layer = inner.layers[index]
        else:
            import gc

            import mlx.nn as nn
            from mlx.utils import tree_flatten, tree_unflatten

            layer = inner.layers[index]
            if page_prefetch is not None:
                page_prefetch.wait(index)
            raw = dict(mx.load(str(paged_manifest.layer_paths[index])))
            local = {}
            for key, value in raw.items():
                mapped_key = remap_gemma4_weight_key(
                    key, layout=model._weetodd_weight_layout
                )
                prefix = f"language_model.model.layers.{index}."
                if mapped_key is not None and mapped_key.startswith(prefix):
                    local[mapped_key.removeprefix(prefix)] = value
            quantized = {
                key.removesuffix(".scales") for key in local if key.endswith(".scales")
            }
            if quantized:
                nn.quantize(
                    layer,
                    group_size=paged_manifest.group_size,
                    bits=paged_manifest.bits,
                    mode="affine",
                    class_predicate=lambda path, _module, names=quantized: path in names,
                )
            expected = {key for key, _value in tree_flatten(layer.parameters())}
            if expected != set(local):
                raise ValueError(
                    f"Paged Gemma layer {index} mismatch: "
                    f"missing={sorted(expected - set(local))[:8]}; "
                    f"unexpected={sorted(set(local) - expected)[:8]}"
                )
            layer.update(tree_unflatten(list(local.items())))
            mx.eval(layer.parameters())
            raw.clear()
            if page_prefetch is not None and index + 1 < total:
                page_prefetch.start(index + 1)
        shared_kv, offset = intermediates[previous_index]
        hidden, shared_kv, offset = layer(
            hidden,
            mask=mask_by_type[layer.layer_type],
            shared_kv=shared_kv,
            offset=offset,
        )
        mx.eval(hidden)
        states.append(hidden)
        intermediates[index] = (shared_kv, offset)
        if progress_callback is not None:
            progress_callback(index + 1, total)
        if paged_manifest is not None:
            local.clear()
            inner.layers[index] = None
            gc.collect()
            mx.clear_cache()
    if paged_manifest is not None:
        object.__setattr__(model, "_weetodd_paged_consumed", True)
    return states, attention_mask


def _remap_feature_weight_key(key: str) -> str | None:
    mappings = (
        ("text_embedding_projection.", "connector.text_embedding_projection."),
        (
            "model.diffusion_model.video_embeddings_connector.",
            "connector.video_embeddings_connector.",
        ),
        (
            "model.diffusion_model.audio_embeddings_connector.",
            "connector.audio_embeddings_connector.",
        ),
        ("video_embeddings_connector.", "connector.video_embeddings_connector."),
        ("audio_embeddings_connector.", "connector.audio_embeddings_connector."),
    )
    for source, target in mappings:
        if key.startswith(source):
            return target + key.removeprefix(source)
    return None


def load_gemma4_feature_extractor(
    path: str | Path,
    *,
    connector_path: str | Path | None = None,
    num_heads: int = 32,
    video_head_dim: int = 128,
    audio_head_dim: int = 64,
    num_connector_layers: int = 8,
    num_registers: int = 128,
    _pack_weights=None,
):
    """Strictly load the dual LTX projection and audiovisual connectors."""
    import mlx.core as mx
    from ltx_core_mlx.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorV2
    from mlx.utils import tree_flatten

    source = Path(path).expanduser()
    inspect_gemma4_pack(source)
    if source.is_dir():
        from .paged_checkpoint import LTX25PagedManifest

        metadata = LTX25PagedManifest.load(source).metadata
    else:
        with safe_open(source, framework="numpy") as handle:
            metadata = handle.metadata() or {}
    raw_config = metadata[GEMMA_CONFIG_METADATA_KEY]
    config: dict[str, Any] = (
        raw_config if isinstance(raw_config, dict) else json.loads(raw_config)
    )
    text_config = config["text_config"]

    pack_weights = _pack_weights if _pack_weights is not None else mx.load(str(source))
    video_projection = pack_weights.get("text_embedding_projection.video_aggregate_embed.weight")
    audio_projection = pack_weights.get("text_embedding_projection.audio_aggregate_embed.weight")
    if video_projection is None or audio_projection is None:
        raise ValueError("LTX 2.5 Gemma pack is missing audiovisual aggregate projections.")
    extractor = GemmaFeaturesExtractorV2(
        caption_channels=int(text_config["hidden_size"]),
        num_gemma_layers=int(text_config["num_hidden_layers"]) + 1,
        video_dim=int(video_projection.shape[0]),
        audio_dim=int(audio_projection.shape[0]),
        num_heads=num_heads,
        video_head_dim=video_head_dim,
        audio_head_dim=audio_head_dim,
        num_connector_layers=num_connector_layers,
        num_registers=num_registers,
    )
    connector_type = _ltx25_connector_class()
    extractor.connector.video_embeddings_connector = connector_type(
        dim=int(video_projection.shape[0]),
        num_heads=num_heads,
        head_dim=video_head_dim,
        num_layers=num_connector_layers,
        num_registers=num_registers,
    )
    extractor.connector.audio_embeddings_connector = connector_type(
        dim=int(audio_projection.shape[0]),
        num_heads=num_heads,
        head_dim=audio_head_dim,
        num_layers=num_connector_layers,
        num_registers=num_registers,
    )
    mapped = {}
    for key, value in pack_weights.items():
        mapped_key = _remap_feature_weight_key(key)
        if mapped_key is not None:
            mapped[mapped_key] = value

    connector_source = Path(connector_path).expanduser() if connector_path else source
    if connector_source != source:
        if connector_source.is_dir():
            from .paged_checkpoint import LTX25PagedManifest

            connector_source = LTX25PagedManifest.load(connector_source).fixed_path
        connector_weights = mx.load(str(connector_source))
        try:
            for key, value in connector_weights.items():
                mapped_key = _remap_feature_weight_key(key)
                if mapped_key is not None:
                    mapped[mapped_key] = value
        finally:
            del connector_weights

    expected = {key for key, _value in tree_flatten(extractor.parameters())}
    actual = set(mapped)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing[:8]))
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected[:8]))
        raise ValueError("Gemma 4 MLX feature-extractor weight mismatch: " + "; ".join(details))
    extractor.load_weights(list(mapped.items()), strict=True)
    extractor.eval()
    return extractor


def _embedded_asset_bytes(path: Path, name: str) -> bytes:
    import numpy as np

    tensor_key = "tokenizer_json" if name == "tokenizer.json" else f"hf_asset__{name}"
    source = path
    metadata = {}
    if path.is_dir():
        from .paged_checkpoint import LTX25PagedManifest

        manifest = LTX25PagedManifest.load(path)
        source = manifest.fixed_path
        metadata = manifest.metadata
    with safe_open(source, framework="numpy") as handle:
        metadata = metadata or handle.metadata() or {}
        if tensor_key in handle.keys():
            value = handle.get_tensor(tensor_key)
            return np.asarray(value).astype(np.uint8, copy=False).tobytes()
        if name in metadata:
            return metadata[name].encode()
    raise ValueError(f"LTX 2.5 Gemma pack {path} is missing embedded asset {name!r}.")


def load_gemma4_tokenizer(path: str | Path, *, max_length: int = 1024):
    """Build the text-only tokenizer from bytes embedded in the Gemma pack."""
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    source = Path(path).expanduser()
    inspect_gemma4_pack(source)
    tokenizer_data = _embedded_asset_bytes(source, "tokenizer.json")
    tokenizer_config = json.loads(_embedded_asset_bytes(source, "tokenizer_config.json"))
    if not isinstance(tokenizer_config, dict):
        raise ValueError("Embedded tokenizer_config.json must contain a JSON object.")
    skipped = {
        "tokenizer_class",
        "auto_map",
        "model_max_length",
        "backend",
        "is_local",
        "local_files_only",
        "processor_class",
        "added_tokens_decoder",
    }
    kwargs = {key: value for key, value in tokenizer_config.items() if key not in skipped}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_buffer(tokenizer_data),
        model_max_length=max_length,
        **kwargs,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Embedded Gemma tokenizer has neither a pad token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        raise ValueError("Embedded Gemma tokenizer has no BOS token.")
    return tokenizer


def tokenize_gemma4(tokenizer, text: str, *, max_length: int = 1024):
    """Tokenize one unpadded, BOS-first prompt for the LTX 2.5 connector."""
    import mlx.core as mx

    encoded = tokenizer(
        text.strip(),
        padding=False,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    token_ids = list(encoded["input_ids"])
    if not token_ids or token_ids[0] != tokenizer.bos_token_id:
        token_ids = [tokenizer.bos_token_id, *token_ids][:max_length]
    mask = [1] * len(token_ids)
    return mx.array([token_ids]), mx.array([mask])


def resolve_prompt_context_length(tokenizer, text: str, mode: str = "official_1024") -> int:
    """Resolve a prompt-token cap; connector registers still extend to 1024."""
    if mode == "official_1024":
        return 1024
    if mode in {"128", "256", "512", "1024"}:
        return int(mode)
    if mode != "auto":
        raise ValueError(f"Unsupported LTX 2.5 prompt context mode: {mode!r}.")

    encoded = tokenizer(text.strip(), padding=False, truncation=False, add_special_tokens=True)
    token_ids = list(encoded["input_ids"])
    required = len(token_ids)
    if not token_ids or token_ids[0] != tokenizer.bos_token_id:
        required += 1
    if required > 1024:
        raise ValueError(
            f"LTX 2.5 prompt requires {required} tokens, exceeding the 1024-token limit."
        )
    return max(128, min(1024, ((required + 127) // 128) * 128))


class LTX25Gemma4Conditioner:
    """Own the packed Gemma 4 and audiovisual connector lifecycle."""

    def __init__(
        self,
        path: str | Path,
        *,
        connector_path: str | Path | None = None,
        max_length: int = 1024,
        feature_kwargs=None,
    ):
        self.path = Path(path).expanduser()
        self.connector_path = (
            Path(connector_path).expanduser() if connector_path is not None else None
        )
        self.max_length = max_length
        self.feature_kwargs = dict(feature_kwargs or {})
        self.model = None
        self.feature_extractor = None
        self.tokenizer = None

    def load(self) -> None:
        if self.model is not None and getattr(self.model, "_weetodd_paged_consumed", False):
            self.free()
        if self.model is not None:
            return
        import mlx.core as mx

        if self.path.is_dir():
            from .paged_checkpoint import LTX25PagedManifest

            pack_weights = mx.load(str(LTX25PagedManifest.load(self.path).fixed_path))
        else:
            pack_weights = mx.load(str(self.path))
        try:
            self.model, _config, _report = load_gemma4_backbone(
                self.path, _pack_weights=pack_weights
            )
            self.feature_extractor = load_gemma4_feature_extractor(
                self.path,
                connector_path=self.connector_path,
                _pack_weights=pack_weights,
                **self.feature_kwargs,
            )
            self.tokenizer = load_gemma4_tokenizer(self.path, max_length=self.max_length)
        except Exception:
            self.free()
            raise
        finally:
            del pack_weights

    def encode(
        self,
        prompt: str,
        *,
        max_length: int | None = None,
        progress_callback=None,
        is_cancelled=None,
    ):
        import mlx.core as mx

        self.load()
        token_ids, attention_mask = tokenize_gemma4(
            self.tokenizer,
            prompt,
            max_length=self.max_length if max_length is None else max_length,
        )
        states, attention_mask = collect_gemma4_hidden_states(
            self.model,
            token_ids,
            attention_mask,
            progress_callback=progress_callback,
            is_cancelled=is_cancelled,
        )
        video, audio = self.feature_extractor(states, attention_mask=attention_mask)
        mx.eval(video, audio)
        return video, audio, attention_mask

    def free(self) -> None:
        if self.model is not None:
            page_prefetch = getattr(self.model, "_weetodd_page_prefetch", None)
            if page_prefetch is not None:
                page_prefetch.close()
        self.model = None
        self.feature_extractor = None
        self.tokenizer = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass
