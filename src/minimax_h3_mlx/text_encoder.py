"""MiniMax-H3's Qwen3-VL-32B conditioner, in MLX.

H3 does not use Qwen3-VL as a language model. It reads the **unnormalized** hidden state after the
50th of its 64 decoder layers (``hidden_states[50]``, where ``hidden_states[0]`` is the embedding
output) and feeds that straight into the DiT's ``condition_proj``. The language-model head, the
final norm and the last 14 decoder layers are never evaluated.

That is worth exploiting: the port loads **only the 50 layers it reads**, skipping ``lm_head``
(151936 x 5120) and layers 50-63 entirely. For a text-only request the vision tower is skipped too.

The transformer stack itself is mlx-vlm's ``qwen3_vl`` implementation — it already has the
interleaved M-RoPE, the ``mrope_section`` split and the deepstack visual merge — so this module only
supplies H3's request presentation, the truncated forward, and a loader that reads a subset.

**Request presentation** (from the reference; no chat template and no special tokens anywhere):
each keyframe contributes a ``"<Picture i>: "`` label followed by a vision block
(``<|vision_start|>``, one ``<|image_pad|>`` per merged patch, ``<|vision_end|>``), then the prompt
verbatim. The rows of a vision block are tagged **video**, not text — that tag is what the DiT's
AdaLN modulation keys off.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open

from .config import TAG_TEXT, TAG_VIDEO
from .packing import TEXT_ENCODER_LAYER


class MiniMaxH3TextEncoder:
    """Qwen3-VL-32B truncated to the layers MiniMax-H3 actually conditions on."""

    def __init__(
        self,
        model_dir: str | Path,
        num_layers: int = TEXT_ENCODER_LAYER,
        dtype: mx.Dtype = mx.bfloat16,
        load_vision: bool = True,
        verbose: bool = False,
        config_path: str | Path | None = None,
        processor_dir: str | Path | None = None,
        tokenizer_dir: str | Path | None = None,
    ):
        from mlx_vlm.models.qwen3_vl.config import ModelConfig, TextConfig, VisionConfig
        from mlx_vlm.models.qwen3_vl.language import Qwen3VLModel
        from mlx_vlm.models.qwen3_vl.vision import VisionModel

        model_dir = Path(model_dir)
        default_config = model_dir / "config.json"
        architecture_config = model_dir / "architecture_config.json"
        if config_path is None and architecture_config.is_file():
            config_path = architecture_config
        with open(config_path or default_config) as fh:
            raw = json.load(fh)

        full_layers = raw["text_config"]["num_hidden_layers"]
        if full_layers <= num_layers:
            raise ValueError(
                f"MiniMax-H3 conditions on hidden_states[{num_layers}] of its Qwen3-VL "
                "conditioner, "
                f"which needs more than {num_layers} decoder layers, but the checkpoint has "
                f"{full_layers}. The last hidden state of a stack truncated to exactly "
                f"{num_layers} "
                "layers is post-norm and is not the conditioning MiniMax-H3 expects."
            )

        self.num_layers = num_layers
        self.full_layers = full_layers
        self.dtype = dtype

        text_raw = dict(raw["text_config"])
        text_raw["num_hidden_layers"] = num_layers  # build only what we evaluate
        self.text_config = TextConfig.from_dict(text_raw)
        self.vision_config = VisionConfig.from_dict(raw["vision_config"])
        self.model_config = ModelConfig.from_dict(
            {
                **raw,
                "text_config": text_raw,
                "vision_config": raw["vision_config"],
                "model_type": raw.get("model_type", "qwen3_vl"),
            }
        )
        self.model_config.text_config = self.text_config
        self.model_config.vision_config = self.vision_config

        self.language = Qwen3VLModel(self.text_config)
        self.vision = VisionModel(self.vision_config) if load_vision else None
        if (model_dir / "paged_text_encoder_manifest.json").is_file():
            if load_vision:
                raise ValueError(
                    "The paged H3 text encoder is text-only. Use the compact resident encoder "
                    "when image conditioning requires Qwen3-VL vision."
                )
            self._load_paged_weights(model_dir)
        else:
            self._load_weights(model_dir, dtype, verbose)

        self.image_token_id = raw["image_token_id"]
        self.vision_start_token_id = raw["vision_start_token_id"]
        self.vision_end_token_id = raw["vision_end_token_id"]
        self.merge_size = self.vision_config.spatial_merge_size

        self._tokenizer = None
        self._processor = None
        self._model_dir = model_dir
        self._processor_dir = Path(processor_dir) if processor_dir is not None else None
        self._tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir is not None else None

    # -- loading ---------------------------------------------------------------------------

    def _wanted(self, key: str) -> tuple[str, str] | None:
        """Map a checkpoint key onto this module's parameter path, or ``None`` to skip it."""
        if key.startswith("lm_head"):
            return None  # never evaluated
        if key.startswith("model.language_model."):
            rest = key[len("model.language_model.") :]
            if rest.startswith("layers."):
                index = int(rest.split(".")[1])
                if index >= self.num_layers:
                    return None  # beyond the conditioning layer
            # `norm` is loaded (it is 5120 floats) to keep the module tree complete, but it is never
            # applied: H3 reads the hidden state *before* the final norm.
            return ("language", rest)
        if key.startswith("model.visual."):
            if self.vision is None:
                return None
            return ("vision", key[len("model.visual.") :])
        # ddalcu's compact MLX export removes the upstream wrappers while retaining the exact
        # module names below them.
        if key.startswith("model."):
            rest = key[len("model.") :]
            if rest.startswith("layers."):
                index = int(rest.split(".")[1])
                if index >= self.num_layers:
                    return None
            return ("language", rest)
        if key.startswith("visual."):
            if self.vision is None:
                return None
            return ("vision", key[len("visual.") :])
        return None

    def _load_weights(self, model_dir: Path, dtype: mx.Dtype, verbose: bool) -> None:
        from mlx.utils import tree_flatten, tree_unflatten

        compact = model_dir / "text_encoder.safetensors"
        shards = (
            [str(compact)]
            if compact.exists()
            else sorted(glob.glob(str(model_dir / "*.safetensors")))
        )
        if not shards:
            raise FileNotFoundError(f"No safetensors in {model_dir}.")

        # Replay the compact export's quantized module structure before calculating expected keys.
        # A sibling ``.scales`` tensor is an unambiguous marker for MLX affine quantization.
        quantized: dict[str, set[str]] = {"language": set(), "vision": set()}
        for shard in shards:
            with safe_open(shard, framework="np") as handle:
                for key in handle.keys():
                    if not key.endswith(".scales"):
                        continue
                    target = self._wanted(key)
                    if target is not None:
                        bucket, path = target
                        quantized[bucket].add(path[: -len(".scales")])

        for bucket, module in (("language", self.language), ("vision", self.vision)):
            if module is not None and quantized[bucket]:
                targets = quantized[bucket]
                nn.quantize(
                    module,
                    group_size=64,
                    bits=8,
                    mode="affine",
                    class_predicate=lambda path, _module, targets=targets: path in targets,
                )

        buckets: dict[str, dict[str, mx.array]] = {"language": {}, "vision": {}}
        expected = {
            "language": {k for k, _ in tree_flatten(self.language.parameters())},
            "vision": (
                set()
                if self.vision is None
                else {k for k, _ in tree_flatten(self.vision.parameters())}
            ),
        }
        # Compact H3 exports omit the final language norm because conditioning is read before it.
        # The module retains its tiny initialized norm for mlx-vlm structural compatibility.
        expected["language"].discard("norm.weight")
        skipped = 0
        for shard in shards:
            for key, tensor in mx.load(shard).items():
                target = self._wanted(key)
                if target is None:
                    skipped += 1
                    continue
                bucket, path = target
                if path not in expected[bucket]:
                    skipped += 1
                    continue
                # Packed MLX weights are uint32 storage. Casting those would silently destroy the
                # checkpoint; floating weights and quantization metadata follow the requested dtype.
                buckets[bucket][path] = (
                    tensor if tensor.dtype == mx.uint32 else tensor.astype(dtype)
                )
            if verbose:
                kept = len(buckets["language"]) + len(buckets["vision"])
                print(f"  {Path(shard).name}: kept {kept}")

        for bucket, module in (("language", self.language), ("vision", self.vision)):
            if module is None:
                continue
            missing = sorted(expected[bucket] - buckets[bucket].keys())
            if missing:
                raise KeyError(
                    f"{bucket} encoder missing {len(missing)} tensors, e.g. {missing[:4]}."
                )
            # mlx-vlm's own layout fixes, chiefly the patch-embed convolution: PyTorch stores it
            # as (out, in, t, h, w) and MLX's conv3d wants (out, t, h, w, in). Loading the bucket
            # raw leaves a transposed kernel that only fails once an image is actually encoded,
            # which is why a text-only path never hit it.
            weights = buckets[bucket]
            sanitize = getattr(module, "sanitize", None)
            if sanitize is not None:
                weights = sanitize(weights)
            module.update(tree_unflatten(list(weights.items())))
        mx.eval(self.language.parameters())
        if self.vision is not None:
            mx.eval(self.vision.parameters())
        self.skipped_tensors = skipped

    def _load_paged_weights(self, model_dir: Path) -> None:
        """Load only fixed text tensors and attach a one-layer sequential executor."""
        from mlx.utils import tree_flatten, tree_unflatten

        from .paged_checkpoint import PagedTensorStore
        from .paged_text_encoder import PagedTextEncoderManifest, PagedTextLayerExecutor

        manifest = PagedTextEncoderManifest.load(model_dir)
        if manifest.num_blocks != self.num_layers:
            raise ValueError(
                f"Paged H3 text encoder has {manifest.num_blocks} layers; "
                f"conditioning requires {self.num_layers}."
            )
        store = PagedTensorStore(manifest)
        source = store.load_fixed()
        fixed: dict[str, mx.array] = {}
        try:
            for key, tensor in source.items():
                target = self._wanted(key)
                if target is None or target[0] != "language":
                    continue
                fixed[target[1]] = tensor
            expected = {
                key
                for key, _ in tree_flatten(self.language.parameters())
                if not key.startswith("layers.") and key != "norm.weight"
            }
            missing = sorted(expected - fixed.keys())
            unexpected = sorted(fixed.keys() - expected)
            if missing or unexpected:
                raise KeyError(
                    f"Paged H3 text fixed tensors mismatch: {len(missing)} missing "
                    f"(e.g. {missing[:4]}), {len(unexpected)} unexpected "
                    f"(e.g. {unexpected[:4]})."
                )
            self.language.update(tree_unflatten(list(fixed.items())))
            self.language.layers = []
            mx.eval(self.language.parameters())
        finally:
            fixed.clear()
            source.clear()
            store.release()
        self.paged_layers = PagedTextLayerExecutor(manifest, self.text_config)
        self.skipped_tensors = 0

    # -- tokenizer / processor -------------------------------------------------------------

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            root = self._model_dir.parent
            path = self._tokenizer_dir
            if path is None:
                path = root / "tokenizer" if (root / "tokenizer").exists() else self._model_dir
            # Current transformers detects the legacy Mistral/Qwen pre-tokenizer regex embedded in
            # this tokenizer and otherwise warns that whitespace/punctuation can split incorrectly.
            self._tokenizer = AutoTokenizer.from_pretrained(str(path), fix_mistral_regex=True)
        return self._tokenizer

    @property
    def processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            directory = self._processor_dir or self._model_dir.parent / "processor"
            if directory.exists():
                self._processor = AutoProcessor.from_pretrained(str(directory))
            else:
                self._processor = _FallbackProcessor(self._build_image_processor())
        return self._processor

    def _build_image_processor(self):
        """Construct Qwen3-VL's image processor from ``vision_config`` alone.

        A compact single-file export carries the weights and the model config but not the upstream
        ``processor/`` directory, and the geometry the processor needs — ``patch_size``,
        ``spatial_merge_size``, ``temporal_patch_size`` — is all present in ``vision_config``.
        Qwen3-VL reuses Qwen2-VL's dynamic-resolution image processor, so the class is the same.

        The only values not derivable from the config are the CLIP normalization statistics, which
        are fixed across the Qwen-VL family, and the pixel budget. The budget is deliberately left
        wide here: keyframes are handed to :meth:`encode` already resized onto the render canvas,
        whose axes are multiples of 32 and therefore survive ``smart_resize`` untouched — one
        vision token per merged 32x32 block, matching the canvas the VAE encodes.
        """
        from transformers import Qwen2VLImageProcessor

        return Qwen2VLImageProcessor(
            patch_size=self.vision_config.patch_size,
            merge_size=self.vision_config.spatial_merge_size,
            temporal_patch_size=self.vision_config.temporal_patch_size,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            min_pixels=32 * 32 * 4,
            max_pixels=32 * 32 * 16384,
            do_resize=True,
            do_rescale=True,
            do_normalize=True,
            do_convert_rgb=True,
        )

    # -- request presentation --------------------------------------------------------------

    def build_request(
        self,
        prompt: str,
        images: list | None = None,
        references: list | None = None,
    ):
        """Build H3's token sequence and its per-row modality tags.

        Returns ``(input_ids, token_tags, vision_inputs)``; ``vision_inputs`` is ``None`` for a
        text-only request, otherwise the processor's ``pixel_values`` / ``image_grid_thw``.
        """
        if not isinstance(prompt, str):
            raise ValueError(f"`prompt` must be a single string, got {type(prompt).__name__}.")

        token_ids: list[int] = []
        token_tags: list[int] = []
        vision_inputs = None

        if images and references:
            raise ValueError(
                "H3 conditioning cannot combine FL2VA keyframes and Ref2VA references."
            )

        if references:
            return self._build_reference_request(prompt, references)

        if images:
            vision = self.processor.image_processor(images=images, return_tensors="np")
            pixel_values = np.asarray(vision["pixel_values"])
            grid_thw = np.asarray(vision["image_grid_thw"])
            merge = self.processor.image_processor.merge_size**2
            start = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            pad = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            end = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")

            for index in range(len(images)):
                num_image_tokens = int(grid_thw[index].prod()) // merge
                label_ids = self.tokenizer(
                    f"<Picture {index + 1}>: ", add_special_tokens=False
                )["input_ids"]
                vision_ids = [start] + [pad] * num_image_tokens + [end]
                token_ids += label_ids + vision_ids
                # The whole vision block is tagged *video*; only the label stays text.
                token_tags += [TAG_TEXT] * len(label_ids) + [TAG_VIDEO] * len(vision_ids)
            vision_inputs = (pixel_values, grid_thw)

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_ids += prompt_ids
        token_tags += [TAG_TEXT] * len(prompt_ids)

        return (
            mx.array(np.array([token_ids], dtype=np.int32)),
            np.array(token_tags, dtype=np.int64),
            vision_inputs,
        )

    def _build_reference_request(self, prompt: str, references: list):
        """Build Ref2VA labels and independently process each ordered visual block."""
        from .ref2va import sample_reference_video_frames

        token_ids: list[int] = []
        token_tags: list[int] = []
        visual_units = []
        counts = {"image": 0, "video": 0, "audio": 0}
        start = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        end = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        image_pad = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        video_pad = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")

        def emit_text(value: str) -> None:
            ids = self.tokenizer(value, add_special_tokens=False)["input_ids"]
            token_ids.extend(ids)
            token_tags.extend([TAG_TEXT] * len(ids))

        def emit_vision(pad_id: int, unit) -> None:
            pixels, grid = unit
            count = int(np.asarray(grid)[0].prod()) // self.merge_size**2
            ids = [start] + [pad_id] * count + [end]
            token_ids.extend(ids)
            token_tags.extend([TAG_VIDEO] * len(ids))
            visual_units.append((pad_id, pixels, np.asarray(grid)))

        for reference in references:
            if reference.has_audio:
                counts["audio"] += 1
                emit_text(f"<Audio {counts['audio']}>: ")
            if reference.kind == "image":
                counts["image"] += 1
                emit_text(f"<Picture {counts['image']}>: ")
                vision = self.processor.image_processor(
                    images=[reference.image], return_tensors="np"
                )
                emit_vision(
                    image_pad,
                    (
                        np.asarray(vision["pixel_values"]),
                        np.asarray(vision["image_grid_thw"]),
                    ),
                )
            elif reference.kind == "video":
                counts["video"] += 1
                emit_text(f"<Video {counts['video']}>: ")
                qwen_frames = (
                    reference.qwen_frames
                    if reference.qwen_frames is not None
                    else reference.frames
                )
                sampled, timestamps = sample_reference_video_frames(qwen_frames)
                reference.block_timestamps = timestamps
                if len(sampled) % 2:
                    sampled.append(sampled[-1])
                for block_index, timestamp in enumerate(timestamps):
                    emit_text(f"<{timestamp:.1f} seconds>")
                    pair = np.stack(sampled[2 * block_index : 2 * block_index + 2])
                    processor = getattr(self.processor, "video_processor", None)
                    if processor is None:
                        raise ValueError(
                            "The selected Qwen3-VL processor has no video processor. "
                            "Use the released H3 processor for Ref2VA."
                        )
                    vision = processor(
                        videos=[pair], do_sample_frames=False, return_tensors="np"
                    )
                    emit_vision(
                        video_pad,
                        (
                            np.asarray(vision["pixel_values_videos"]),
                            np.asarray(vision["video_grid_thw"]),
                        ),
                    )
        emit_text(prompt)
        return (
            mx.array(np.asarray([token_ids], dtype=np.int32)),
            np.asarray(token_tags, dtype=np.int64),
            visual_units,
        )

    # -- forward ---------------------------------------------------------------------------

    def _hidden_states(
        self,
        input_ids: mx.array,
        position_ids: mx.array,
        inputs_embeds: mx.array | None = None,
        visual_pos_masks: mx.array | None = None,
        deepstack_visual_embeds: list | None = None,
    ) -> mx.array:
        """Run the truncated stack and return the hidden state **before** the final norm."""
        from mlx_vlm.models.base import create_attention_mask

        model = self.language
        h = model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        mask = create_attention_mask(h, None)

        position_embeddings = None
        paged = getattr(self, "paged_layers", None)
        if paged is None:
            if position_ids is not None and not model.layers[0].self_attn.rotary_emb.fused_apply:
                position_embeddings = model.layers[0].self_attn.rotary_emb(h, position_ids)

            for layer_idx, layer in enumerate(model.layers):
                h = layer(h, mask, None, position_ids, position_embeddings)
                if deepstack_visual_embeds is not None and layer_idx < len(
                    deepstack_visual_embeds
                ):
                    h = model._deepstack_process(
                        h, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                    )
        else:
            for layer_idx in range(paged.num_layers):
                with paged.layer(layer_idx) as layer:
                    if (
                        layer_idx == 0
                        and position_ids is not None
                        and not layer.self_attn.rotary_emb.fused_apply
                    ):
                        position_embeddings = layer.self_attn.rotary_emb(h, position_ids)
                        mx.eval(position_embeddings)
                    h = layer(h, mask, None, position_ids, position_embeddings)
                    if deepstack_visual_embeds is not None and layer_idx < len(
                        deepstack_visual_embeds
                    ):
                        h = model._deepstack_process(
                            h, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                        )
                    # Complete the sequential state before the layer's mapped weights are retired.
                    mx.eval(h)
        # No `model.norm(h)`: H3 conditions on the unnormalized state.
        return h

    def encode(
        self,
        prompt: str,
        images: list | None = None,
        references: list | None = None,
    ) -> tuple[mx.array, np.ndarray]:
        """Encode a request into ``((1, num_text_tokens, 5120), (num_text_tokens,))``."""
        from mlx_vlm.models.qwen3_vl.language import LanguageModel

        input_ids, token_tags, vision_inputs = self.build_request(prompt, images, references)

        inputs_embeds = None
        visual_pos_masks = None
        deepstack_embeds = None
        image_grid_thw = None
        video_grid_thw = None

        if references:
            if self.vision is None:
                raise ValueError(
                    "This encoder was built with `load_vision=False`; it cannot take references."
                )
            features = []
            deepstack_groups = []
            image_grids = []
            video_grids = []
            for pad_id, pixels, grid_np in vision_inputs:
                grid = mx.array(grid_np.astype(np.int32))
                hidden, deep = self.vision(
                    mx.array(pixels).astype(self.dtype), grid, output_hidden_states=True
                )
                features.append(hidden.astype(self.dtype))
                deepstack_groups.append(deep)
                if pad_id == self.image_token_id:
                    image_grids.append(grid_np)
                else:
                    video_grids.append(grid_np)
            inputs_embeds = self.language.embed_tokens(input_ids)
            visual_mask = (input_ids == self.image_token_id) | (
                input_ids == self.config.video_token_id
            )
            combined = mx.concatenate(features, axis=0)
            expanded = mx.broadcast_to(visual_mask[..., None], inputs_embeds.shape)
            if int(expanded.sum().item()) != combined.size:
                raise ValueError(
                    "The Ref2VA presentation rows do not match the Qwen3-VL vision features."
                )
            inputs_embeds = _masked_scatter(inputs_embeds, expanded, combined)
            visual_pos_masks = visual_mask
            if deepstack_groups:
                deepstack_embeds = [
                    mx.concatenate([group[layer] for group in deepstack_groups], axis=0)
                    for layer in range(len(deepstack_groups[0]))
                ]
            if image_grids:
                image_grid_thw = mx.array(np.concatenate(image_grids).astype(np.int32))
            if video_grids:
                video_grid_thw = mx.array(np.concatenate(video_grids).astype(np.int32))
        elif vision_inputs is not None:
            if self.vision is None:
                raise ValueError(
                    "This encoder was built with `load_vision=False`; it cannot take images."
                )
            pixel_values, grid_np = vision_inputs
            image_grid_thw = mx.array(grid_np.astype(np.int32))
            hidden, deepstack_embeds = self.vision(
                mx.array(pixel_values).astype(self.dtype),
                image_grid_thw,
                output_hidden_states=True,
            )
            inputs_embeds = self.language.embed_tokens(input_ids)
            image_mask = input_ids == self.image_token_id
            # The vision tower emits one row per *merged* patch, not one per request token, so the
            # rows have to be scattered into the `<|image_pad|>` positions. A `where` cannot do it:
            # its operands would have to broadcast, and (1, num_patches, 5120) does not broadcast
            # against (1, sequence_length, 5120) for any request that carries prompt text as well.
            expanded = mx.broadcast_to(image_mask[..., None], inputs_embeds.shape)
            image_features = hidden.astype(inputs_embeds.dtype)
            if int(expanded.sum().item()) != image_features.size:
                raise ValueError(
                    f"The request reserves {int(image_mask.sum().item())} image rows but the "
                    f"vision tower produced {image_features.shape[0]}. The image processor's "
                    "patch geometry and the vision config disagree."
                )
            inputs_embeds = _masked_scatter(inputs_embeds, expanded, image_features)
            visual_pos_masks = image_mask

        # Qwen3-VL's 3D M-RoPE index, derived from the vision-start/pad token ids.
        position_ids, _ = LanguageModel.get_rope_index(
            self,
            input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=None,
        )

        hidden_states = self._hidden_states(
            input_ids,
            position_ids,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_embeds,
        )
        mx.eval(hidden_states)
        return hidden_states, token_tags

    # `LanguageModel.get_rope_index` reads `self.config`; expose the same attribute.
    @property
    def config(self):
        return self.model_config


class _FallbackProcessor:
    """The single attribute :meth:`build_request` needs, when no upstream processor dir exists."""

    def __init__(self, image_processor):
        self.image_processor = image_processor


def _masked_scatter(target: mx.array, mask: mx.array, values: mx.array) -> mx.array:
    """Write ``values`` into the ``True`` positions of ``mask``, in flattened order.

    The same operation mlx-vlm performs when merging vision rows into a Qwen3-VL request; kept
    local so this module does not depend on a private helper of theirs.
    """
    shape = target.shape
    flat = mx.flatten(target)
    positions = mx.array(np.where(np.array(mx.flatten(mask)))[0], mx.uint32)
    flat[positions] = mx.flatten(values)
    return mx.reshape(flat, shape)
