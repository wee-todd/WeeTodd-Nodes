"""Local asset references without copying weights or claiming inferred compatibility.

Registry IDs identify local assets, not content hashes. Revisions detect ordinary
file changes using stat/header/config evidence; they are not full payload hashes.
Engine-specific preflight remains authoritative for supported runtime layouts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import uuid
from pathlib import Path

from .model_library import inspect_safetensors_header

FORMAT = "weetodd-asset-registry-v1"
MAX_CONFIG_BYTES = 32 * 1024 * 1024
MAX_ASSET_FILES = 10000
DATA_SUFFIXES = {".safetensors", ".json", ".txt", ".model", ".tiktoken"}
COMPONENT_FIELDS = {
    "h3": {
        "checkpoint",
        "transformer",
        "text_encoder",
        "processor",
        "tokenizer",
        "video_vae",
        "audio_vae",
    },
    "ltx23": {"model_dir", "gemma_model"},
    "ltx25": {
        "transformer_path",
        "text_encoder_path",
        "video_vae_path",
        "audio_vae_path",
        "spatial_upscaler_path",
        "duration_head_path",
        "distilled_lora_path",
        "msr_lora_path",
    },
}


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _stat(path):
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns]


def describe_asset(path, *, manifest_only=False):
    """Read headers and small support files, never materialize weight payloads.

    Directory member names are part of a runtime layout; the root name is not.
    The H3 checkpoint root supplies only the model manifest when all components
    are bound separately, avoiding inspection of unselected weight alternatives.
    """
    root = Path(path).expanduser().absolute()
    if not root.exists():
        raise FileNotFoundError(f"Missing local asset: {root}")
    if manifest_only:
        files = [root / "model_index.json"]
    elif root.is_file():
        files = [root]
    else:
        files, visited = [], set()

        def walk_error(error):
            raise error

        for folder, directories, names in os.walk(root, followlinks=True, onerror=walk_error):
            folder = Path(folder)
            identity = tuple(_stat(folder)[:2])
            if identity in visited:
                directories[:] = []
                continue
            visited.add(identity)
            directories[:] = sorted(d for d in directories if d not in {".git", ".cache"})
            files.extend(
                folder / n
                for n in sorted(names)
                if not n.startswith("._") and Path(n).suffix.lower() in DATA_SUFFIXES
            )
            if len(files) > MAX_ASSET_FILES:
                raise ValueError("Asset contains too many files; select a component directory")
    if not files:
        raise ValueError(f"No inspectable local component files: {root}")
    # A single present shard is not a complete checkpoint. These conventional
    # shard names describe storage, never architecture or trained task identity.
    shards = {}
    for file in files:
        match = re.fullmatch(r"(.+)-(\d+)-of-(\d+)\.safetensors", file.name)
        if match:
            stem, number, count = match.groups()
            number, count = int(number), int(count)
            if not 1 <= number <= count <= MAX_ASSET_FILES:
                raise ValueError(f"Invalid shard numbering: {file}")
            group = shards.setdefault((file.parent, stem), {"count": count, "numbers": set()})
            if group["count"] != count or number in group["numbers"]:
                raise ValueError(f"Conflicting shard numbering: {file}")
            group["numbers"].add(number)
    if any(len(group["numbers"]) != group["count"] for group in shards.values()):
        raise ValueError(f"Incomplete sharded checkpoint: {root}")
    members = []
    for file in sorted(files):
        before = _stat(file)
        member = {"name": str(file.relative_to(root)) if root.is_dir() else ".", "stat": before}
        if file.suffix.lower() == ".safetensors":
            header = inspect_safetensors_header(file, include_tensors=True)
            member.update(
                tensor_count=header["tensor_count"],
                dtypes=header["dtypes"],
                metadata=header["metadata"],
                tensor_layout_sha256=_digest(header["tensors"]),
            )
        elif file.suffix.lower() in DATA_SUFFIXES:
            if before[2] > MAX_CONFIG_BYTES:
                raise ValueError(f"Support file exceeds bounded inspection size: {file}")
            content = file.read_bytes()
            member["sha256"] = hashlib.sha256(content).hexdigest()
            if file.name.endswith(".safetensors.index.json"):
                weight_map = json.loads(content).get("weight_map")
                if not isinstance(weight_map, dict) or not weight_map:
                    raise ValueError(f"Invalid safetensors shard index: {file}")
                missing_shards = set()
                for tensor_name, shard in weight_map.items():
                    if (
                        not isinstance(shard, str)
                        or Path(shard).is_absolute()
                        or ".." in Path(shard).parts
                    ):
                        raise ValueError(f"Invalid indexed shard for {tensor_name}: {shard}")
                    if not (file.parent / shard).is_file():
                        missing_shards.add(shard)
                if missing_shards:
                    # Some MLX conversions retain an upstream shard index while
                    # the active loader discovers the converted files by glob.
                    # Only the engine can decide whether this index is binding.
                    member["unresolved_index_files"] = sorted(missing_shards)
        else:
            raise ValueError(f"Unsupported asset format; conversion is required: {file.suffix}")
        if _stat(file) != before:
            raise ValueError(f"Asset changed during inspection: {file}")
        members.append(member)
    structural = [{k: v for k, v in m.items() if k != "stat"} for m in members]
    return {
        "kind": "directory" if root.is_dir() else "file",
        "physical_identity": _stat(root)[:2],
        "manifest_only": manifest_only,
        "members": members,
        "revision": _digest(members),
        "structural_sha256": _digest(structural),
        "payload_hash_verified": False,
        "compatibility": "not_evaluated",
    }


class AssetRegistry:
    """Small JSON registry with atomic writes and a cooperative writer lock."""

    def __init__(self, path):
        self.path = Path(path).expanduser().absolute()
        self.original = self.path.read_bytes() if self.path.exists() else None
        self.data = json.loads(self.original) if self.original else {"format": FORMAT, "assets": {}}
        if self.data.get("format") != FORMAT or not isinstance(self.data.get("assets"), dict):
            raise ValueError("Unsupported model asset registry")

    def register(self, path, *, manifest_only=False):
        path = str(Path(path).expanduser().absolute())
        if (
            not manifest_only
            and Path(path).is_dir()
            and self.path.resolve().is_relative_to(Path(path).resolve())
        ):
            raise ValueError("Keep the registry outside selected component directories")
        descriptor = describe_asset(path, manifest_only=manifest_only)
        identity = descriptor["physical_identity"]
        match = next(
            (
                key
                for key, value in self.data["assets"].items()
                if value["descriptor"]["physical_identity"] == identity
                and value["descriptor"]["manifest_only"] == manifest_only
            ),
            None,
        )
        asset_id = match or "asset-" + uuid.uuid4().hex
        previous = self.data["assets"].get(asset_id, {})
        aliases = list(dict.fromkeys([path, *previous.get("paths", [])]))
        self.data["assets"][asset_id] = {"paths": aliases, "descriptor": descriptor}
        return {"asset_id": asset_id, "revision": descriptor["revision"], "path_hint": path}

    def resolve(self, reference):
        if set(reference) != {"asset_id", "revision", "path_hint"}:
            raise ValueError("Asset references require asset_id, revision, and path_hint")
        asset = self.data["assets"].get(reference["asset_id"])
        if asset is None:
            raise ValueError(f"Unknown model asset: {reference['asset_id']}")
        if reference["revision"] != asset["descriptor"]["revision"]:
            raise ValueError("Asset revision changed; review and re-import the recipe")
        problems = []
        paths = list(dict.fromkeys([reference["path_hint"], *asset["paths"]]))
        for path in paths:
            try:
                descriptor = describe_asset(
                    path, manifest_only=asset["descriptor"]["manifest_only"]
                )
                if descriptor["revision"] != reference["revision"]:
                    raise ValueError("Asset changed since import; review and re-import it")
                return path, descriptor
            except (OSError, ValueError) as exc:
                problems.append(str(exc))
        raise ValueError("No unchanged local path for asset: " + "; ".join(problems))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_name(self.path.name + ".lock")
        # Do not remove another process's lock if exclusive creation fails.
        with lock.open("x"):
            try:
                current = self.path.read_bytes() if self.path.exists() else None
                if current != self.original:
                    raise ValueError("Registry changed concurrently; reload before saving")
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=self.path.parent, prefix=".asset-registry-", delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    try:
                        json.dump(self.data, stream, indent=2, allow_nan=False)
                        stream.write("\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    except BaseException:
                        temporary.unlink(missing_ok=True)
                        raise
                try:
                    os.replace(temporary, self.path)
                finally:
                    temporary.unlink(missing_ok=True)
                self.original = self.path.read_bytes()
            finally:
                lock.unlink()


def _slots(recipe):
    """Enumerate explicit model inputs, never prompt/media/output strings."""
    engine = recipe["engine"]
    for key in sorted(COMPONENT_FIELDS[engine]):
        if recipe["components"].get(key):
            yield recipe["components"], key, "components." + key
    if engine in {"h3", "ltx23"}:
        for i, adapter in enumerate(recipe.get("loras", {}).get("adapters", [])):
            for key in ("path", "adaln_input_grid"):
                if adapter.get(key):
                    yield adapter, key, f"loras.adapters.{i}.{key}"
    if engine == "ltx23":
        for name in ("loras", "ic_loras"):
            for i, adapter in enumerate(recipe["components"].get(name, [])):
                yield adapter, "path", f"components.{name}.{i}.path"
    if engine == "h3":
        for key in (
            "repository",
            "checkpoint",
            "model_spec",
            "linear_branch",
            "default_adapter",
            "turbo_adapter",
        ):
            if recipe.get("vdn", {}).get(key):
                yield recipe["vdn"], key, "vdn." + key
    if engine == "ltx25":
        for key in ("loras", "ic_loras"):
            for i, adapter in enumerate(recipe["components"].get(key, [])):
                yield adapter, 0, f"components.{key}.{i}.path"
        for key in (
            "dfr_detailing_lora_path",
            "dfr_prebaked_transformer_path",
            "dfr_temporal_upsampler_path",
        ):
            if recipe.get("config", {}).get(key):
                yield recipe["config"], key, "config." + key


def import_recipe(recipe, registry):
    """Bind existing model inputs; engine preflight must run before accepting import."""
    recipe = copy.deepcopy(recipe)
    if recipe["engine"] == "h3":
        from wee_todd_nodes.preflight import H3ComponentSetSpec

        fields = dict(recipe["components"])
        fields.pop("preview_override", None)
        paths = H3ComponentSetSpec(**fields).resolved_paths()
        recipe["components"].update({name: str(path) for name, path in paths.items()})
        # Preserve the *existing* recipe's math when an alias is later relocated.
        # Legacy auto detection is filename-sensitive; bind its resolved values
        # explicitly, without claiming this is a new content-based detector.
        from wee_todd_nodes.lora import H3LoRASpec

        for adapter in recipe.get("loras", {}).get("adapters", []):
            spec = H3LoRASpec(**adapter)
            adapter["profile"] = spec.resolved_profile
            adapter["qkv_layout"] = spec.resolved_qkv_layout
    for parent, key, role in _slots(recipe):
        if not isinstance(parent[key], str):
            raise ValueError("Import expects resolved local paths, not existing asset references")
        parent[key] = registry.register(parent[key], manifest_only=role == "components.checkpoint")
    return recipe


def resolve_recipe(recipe, registry=None):
    recipe = copy.deepcopy(recipe)
    assets = []
    for parent, key, role in _slots(recipe):
        if isinstance(parent[key], dict):
            if registry is None:
                raise ValueError("Recipe uses asset references; supply --model-library")
            reference = parent[key]
            path, descriptor = registry.resolve(reference)
            parent[key] = path
            assets.append(
                {
                    "role": role,
                    **reference,
                    "path": path,
                    "structural_sha256": descriptor["structural_sha256"],
                    "payload_hash_verified": False,
                    "warnings": [
                        {
                            "index": member["name"],
                            "unresolved_index_files": member["unresolved_index_files"],
                            "action": "Engine must determine whether this index is active",
                        }
                        for member in descriptor["members"]
                        if member.get("unresolved_index_files")
                    ],
                }
            )
        elif not isinstance(parent[key], str):
            raise ValueError(f"Invalid asset input at {role}")
    return recipe, {
        "format": "weetodd-asset-resolution-v1",
        "assets": assets,
        "copied_weight_bytes": 0,
        "downloaded_bytes": 0,
        "qualification": "not_evaluated",
        "identity_note": "Local IDs and stat/header revisions, not payload hashes",
    }
