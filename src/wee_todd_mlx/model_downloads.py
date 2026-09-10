"""Explicit, pinned model preparation shared by Studio and command-line setup.

No network or weighted imports occur at import time. Preparation never replaces
an existing model, and incomplete downloads remain reusable after cancellation.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DownloadFile:
    repo: str
    revision: str
    filename: str
    size: int
    sha256: str
    target: str
    provider: str = "huggingface"

    def __post_init__(self):
        if self.provider not in {"huggingface", "github"}:
            raise ValueError("Unsupported download provider")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", self.repo):
            raise ValueError("Invalid model repository")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Model downloads require a pinned revision")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256) or self.size <= 0:
            raise ValueError("Model downloads require an exact size and SHA-256")
        for name in (self.filename, self.target):
            if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
                raise ValueError("Download filename must remain inside its model directory")

    @property
    def url(self):
        if self.provider == "github":
            return f"https://raw.githubusercontent.com/{self.repo}/{self.revision}/{self.filename}"
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{urllib.parse.quote(self.filename)}"


COMPACT_REPO = "ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit"
COMPACT_REVISION = "64314cde0ac6d90f132bc94ae58e0c82f77396c6"
QWEN_FILES = (
    DownloadFile(
        COMPACT_REPO,
        COMPACT_REVISION,
        "text_encoder.safetensors",
        28222741760,
        "79514b061aa0acbeb802bbe37e4853f1b0d0e6fe7fad4f61624057201773bf56",
        "compact/text_encoder.safetensors",
    ),
    DownloadFile(
        COMPACT_REPO,
        COMPACT_REVISION,
        "config.json",
        724,
        "add27928e87c54c6a88901d8a8c177df47691661e455f6747bc8916cee6149ec",
        "compact/config.json",
    ),
    DownloadFile(
        "Qwen/Qwen3-VL-32B-Instruct",
        "0cfaf48183f594c314753d30a4c4974bc75f3ccb",
        "config.json",
        1474,
        "d2dd0c60d01b9e195d9447c52da61c7302d28828524914c044d9c6e1b81d0427",
        "architecture/config.json",
    ),
)

QWEN_FILES += (
    DownloadFile(
        COMPACT_REPO,
        COMPACT_REVISION,
        "LICENSE",
        17604,
        "59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44",
        "compact/LICENSE",
    ),
    DownloadFile(
        COMPACT_REPO,
        COMPACT_REVISION,
        "NOTICE",
        552,
        "3ea3c60ee64954fc0898e68b7534e3613ba74586256aec6edad557b7f217c7c6",
        "compact/NOTICE",
    ),
    DownloadFile(
        COMPACT_REPO,
        COMPACT_REVISION,
        "MODIFICATIONS.md",
        1630,
        "3439a033d427a8b3225640fad4dc76787496982ee4565deafc556852a7af902a",
        "compact/MODIFICATIONS.md",
    ),
)
LTX25_REVISION = "5e6e71018ee1756ed329b697a7b4aedc934dfce9"
LTX25_FILES = (
    DownloadFile(
        "Lightricks/LTX-2.5",
        LTX25_REVISION,
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        42018190584,
        "31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4",
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    ),
    DownloadFile(
        "Lightricks/LTX-2.5",
        LTX25_REVISION,
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        995778752,
        "eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8",
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    ),
    DownloadFile(
        "Lightricks/LTX-2.5",
        LTX25_REVISION,
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        26263858182,
        "ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    ),
    DownloadFile(
        "Lightricks/LTX-2.5",
        LTX25_REVISION,
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
        364866540,
        "c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5",
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
    ),
    DownloadFile(
        "Lightricks/LTX-2.5",
        LTX25_REVISION,
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        1452269922,
        "685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d",
        "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    ),
    DownloadFile(
        "Lightricks/LTX-2",
        "a95ab856bf29407b6b066ede0abe1846050db56c",
        "LICENSE-2_x",
        30399,
        "be75acae5c99b0fb16ed6cfbf8f731e5121a729bef112d20337699407e796451",
        "LICENSE-2_x",
        provider="github",
    ),
)

RESERVE_BYTES = 256 * 1024 * 1024
PRECONVERTED = json.loads(Path(__file__).with_name("model_download_catalog.json").read_text())


def download_catalog():
    size = sum(item.size for item in QWEN_FILES)
    result = [
        {
            "id": "h3-qwen-q8-vision",
            "name": "H3 Q8 vision encoder · Convert from source",
            "description": "Download the compact Q8 encoder and prepare vision-capable pages. "
            "For H3 text, first/last-frame and reference clips. "
            "Transformer, VAEs and tokenizer/processor are separate components. "
            "The verified source download is retained for reuse.",
            "downloadBytes": size,
            "requiredDiskBytes": size * 2 + RESERVE_BYTES,
            "sourceURL": f"https://huggingface.co/{COMPACT_REPO}/tree/{COMPACT_REVISION}",
            "licenseURL": f"https://huggingface.co/{COMPACT_REPO}/blob/{COMPACT_REVISION}/LICENSE",
            "outputKind": "directory",
            "engines": ["h3"],
            "components": ["text_encoder"],
            "licenseNotice": "The source bundle includes MiniMax H3 territorial restrictions "
            "(including the U.S., EU, UK and Republic of Korea) and separate "
            "Qwen attribution. Review the upstream terms for your use before download.",
        }
    ]

    ltx_size = sum(item.size for item in LTX25_FILES)
    result.append(
        {
            "id": "ltx25-distilled-q8",
            "name": "LTX 2.5 distilled Q8 · Convert from source",
            "description": "Download five official distilled components and convert transformer "
            "and Gemma encoder to Q8 pages. Includes the convolutional video VAE, "
            "audio VAE and spatial upscaler. Verified source files are retained.",
            "downloadBytes": ltx_size,
            "requiredDiskBytes": ltx_size * 2 + RESERVE_BYTES,
            "sourceURL": f"https://huggingface.co/Lightricks/LTX-2.5/tree/{LTX25_REVISION}",
            "licenseURL": "https://github.com/Lightricks/LTX-2/blob/a95ab856bf29407b6b066ede0abe1846050db56c/LICENSE-2_x",
            "licenseNotice": "Accept the LTX terms on Hugging Face before downloading. "
            "Setup uses a Studio Keychain token or your existing Hugging Face login.",
            "outputKind": "directory",
            "engines": ["ltx25"],
            "components": [
                "transformer_path",
                "text_encoder_path",
                "video_vae_path",
                "audio_vae_path",
                "spatial_upscaler_path",
            ],
        }
    )
    return [dict(item["descriptor"]) for item in PRECONVERTED] + result


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def partial_path(item, target):
    return target.with_name(f".{target.name}.{item.sha256}.part")


def _origin(url):
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        if result and _origin(newurl) != _origin(req.full_url):
            result.remove_header("Authorization")
        if result and urllib.parse.urlparse(newurl).scheme != "https":
            raise ValueError("Refusing an insecure model download redirect")
        return result


def _open(request, *, timeout):
    return urllib.request.build_opener(_SafeRedirect()).open(request, timeout=timeout)


def _hf_token():
    # Reuse an existing login; never persist credentials in recipes or setup logs.
    try:
        from huggingface_hub import get_token

        return get_token()
    except ImportError:
        return os.environ.get("HF_TOKEN")


def download_file(item, target, *, opener=None, progress=None):
    """Bounded streaming with validated ranges and full-file integrity before rename."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    report = progress or (lambda *_: None)
    if target.exists():
        report(f"Checking existing {item.filename}", 0.0)
        if target.stat().st_size == item.size and _hash(target) == item.sha256:
            report(f"Using verified {item.filename}", 1.0)
            return target
        raise ValueError(
            f"An existing download differs from its checksum: {target}. "
            "Move that file aside before retrying."
        )
    part = partial_path(item, target)
    offset = part.stat().st_size if part.exists() else 0
    if offset > item.size:
        raise ValueError(f"Partial download is larger than expected: {part}")
    if offset < item.size:
        headers = {"User-Agent": "WeeTodd-Model-Setup/1", "Accept-Encoding": "identity"}
        if item.provider == "huggingface" and (token := _hf_token()):
            headers["Authorization"] = "Bearer " + token
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(item.url, headers=headers)
        try:
            response = (opener or _open)(request, timeout=30)
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                raise ValueError(
                    "Model access requires accepting its source terms and signing "
                    "in through Studio’s Hugging Face access controls or hf auth login, "
                    "then retrying setup."
                ) from None
            raise ValueError(
                f"Model download unavailable (HTTP {error.code}); retry setup."
            ) from None
        with response:
            if response.status == 206:
                expected = f"bytes {offset}-{item.size - 1}/{item.size}"
                if response.headers.get("Content-Range") != expected:
                    raise ValueError("Download range does not match the requested file bytes")
            elif response.status == 200:
                offset = (
                    0  # A server may ignore Range; restart instead of appending duplicate bytes.
                )
            else:
                raise ValueError(f"Unexpected download status {response.status}")
            mode = "ab" if offset else "wb"
            report(f"Downloading {item.filename}", offset / item.size)
            updated = time.monotonic()
            with part.open(mode) as stream:
                while chunk := response.read(4 * 1024 * 1024):
                    if offset + len(chunk) > item.size:
                        raise ValueError("Download exceeds its pinned file size")
                    stream.write(chunk)
                    offset += len(chunk)
                    if time.monotonic() - updated >= 0.5:
                        report(f"Downloading {item.filename}", offset / item.size)
                        updated = time.monotonic()
                stream.flush()
                os.fsync(stream.fileno())
        if offset != item.size:
            raise ValueError("Download incomplete; retry setup to resume the partial file")
    report(f"Verifying {item.filename}", 1.0)
    if _hash(part) != item.sha256:
        part.unlink()
        raise ValueError("Download SHA-256 verification failed; retry to download a clean copy")
    # The setup lock serializes writers. Refuse replacement even for direct helper callers.
    os.link(part, target)
    part.unlink()
    return target


def _check_space(destination, required):
    free = shutil.disk_usage(destination).free
    if free < required:
        raise ValueError(
            f"Model setup needs {required / 1e9:.1f} GB of free disk space; "
            f"{free / 1e9:.1f} GB is available. Choose another model folder."
        )


def _convert_qwen(source, destination, architecture):
    from minimax_h3_mlx.paged_text_encoder import convert_to_paged_text_encoder

    manifest = convert_to_paged_text_encoder(
        source,
        destination,
        architecture_config=architecture,
        include_vision=True,
        verify_output=True,
    )
    if not manifest.supports_vision:
        raise ValueError("Prepared encoder is missing its vision page")


def _convert_ltx25(source, destination, progress):
    from ltx25_mlx.paged_checkpoint import convert_to_paged_q8
    from ltx25_mlx.runtime import LTX25ComponentSpec

    destination.mkdir()
    transformer = "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    gemma = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
    for index, (kind, relative) in enumerate((("transformer", transformer), ("gemma", gemma))):
        progress(f"Converting LTX 2.5 {kind} to verified Q8 pages…", 0.65 + index * 0.14)
        convert_to_paged_q8(source / relative, destination / kind, kind=kind, verify_output=True)
    for item in LTX25_FILES:
        if item.filename in {transformer, gemma}:
            continue
        target = destination / item.target
        target.parent.mkdir(parents=True, exist_ok=True)
        # Same-volume immutable verified sources can share bytes without duplicating disk usage.
        try:
            os.link((source / item.target).resolve(), target)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copy2(source / item.target, target)
    progress("Validating prepared LTX 2.5 components…", 0.96)
    LTX25ComponentSpec(
        transformer_path=str(destination / "transformer"),
        text_encoder_path=str(destination / "gemma"),
        video_vae_path=str(destination / "vae/ltx-2.5-video-vae-conv-bf16.safetensors"),
        audio_vae_path=str(destination / "vae/ltx-2.5-audio-vae-bf16.safetensors"),
        spatial_upscaler_path=str(
            destination
            / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
        ),
    ).validate("distilled")


def reuse_sources(files, cache, roots, progress=None):
    """Reference exact existing source bytes; names alone never establish identity."""
    report = progress or (lambda *_: None)
    missing = [item for item in files if not (cache / item.target).exists()]
    seen = set()
    pending = [Path(root).expanduser() for root in roots]
    inspected = 0
    while pending and missing:
        candidate = pending.pop()
        inspected += 1
        if inspected > 50000:
            report("Source reuse scan reached its limit; choose a more specific model folder.", 0.0)
            break
        try:
            stat = candidate.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            if candidate.is_dir():
                with os.scandir(candidate) as entries:
                    for entry in entries:
                        if not entry.name.startswith("."):
                            pending.append(Path(entry.path))
                        if len(pending) > 50000:
                            raise ValueError("Too many source files; choose a more specific folder")
                continue
            matches = [item for item in missing if item.size == stat.st_size]
            if not matches or not candidate.is_file():
                continue
            report(f"Checking reusable source {candidate.name}…", 0.0)
            digest = _hash(candidate)
            for item in matches:
                if digest != item.sha256:
                    continue
                target = cache / item.target
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(candidate.resolve(), target)
                except OSError as error:
                    if error.errno != errno.EXDEV:
                        raise
                    target.symlink_to(candidate.resolve())
                missing.remove(item)
                report(f"Reusing verified {candidate.name}", 0.0)
        except (FileNotFoundError, PermissionError):
            continue


def publish_directory(source, destination):
    """Atomically install a directory without replacing even an empty destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        result = rename(os.fsencode(source), os.fsencode(destination), 0x4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        result = rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    else:
        raise OSError(
            "This platform cannot atomically publish model directories without replacement"
        )
    if result:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), str(destination))


def _validate_preconverted(kind, root):
    if kind == "h3-qwen":
        from minimax_h3_mlx.paged_text_encoder import PagedTextEncoderManifest

        if not PagedTextEncoderManifest.load(root).supports_vision:
            raise ValueError("Downloaded encoder lacks its required vision page")
    elif kind == "h3-dt-tokenizer":
        from .model_setup import _h3_candidate

        _h3_candidate("tokenizer", root, "t2va")
    elif kind in {
        "h3-transformer-fl2va",
        "h3-transformer-ref2va",
        "h3-video-vae",
        "h3-support-fl2va",
        "h3-support-ref2va",
    }:
        _validate_h3_preconverted(kind, root)
    elif kind == "ltx25":
        from ltx25_mlx.runtime import LTX25ComponentSpec

        LTX25ComponentSpec(
            transformer_path=str(root / "transformer"),
            text_encoder_path=str(root / "gemma"),
            video_vae_path=str(root / "vae/ltx-2.5-video-vae-conv-bf16.safetensors"),
            audio_vae_path=str(root / "vae/ltx-2.5-audio-vae-bf16.safetensors"),
            spatial_upscaler_path=str(
                root
                / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
            ),
        ).validate("distilled")
    else:
        raise ValueError("Unknown preconverted component kind")


def _validate_h3_preconverted(kind, root):
    from .model_library import inspect_safetensors_header
    from .model_setup import _h3_candidate, _json, _object

    task = "ref2va" if kind.endswith("ref2va") else "fl2va"
    if kind.startswith("h3-transformer-"):
        # Shapes identify the shared H3 architecture, not FL2VA/Ref2VA training
        # identity. Exact payload identity comes from the pinned download catalog.
        _json(root / "paged_manifest.json")
        for filename in root.rglob("*.safetensors"):
            inspect_safetensors_header(filename)
        _h3_candidate("transformer", root, task)
    elif kind == "h3-video-vae":
        filename = root / "video_vae_affine_q8.safetensors"
        header = inspect_safetensors_header(filename)
        wrapper = _object(
            json.loads(header["metadata"].get("minimax_h3_video_vae", "{}")),
            "video_vae metadata",
        )
        if wrapper.get("format") != "minimax-h3-mlx-video-vae":
            raise ValueError("Downloaded video_vae metadata lacks the native MLX format")
        _h3_candidate("video_vae", filename, task)
    else:
        _h3_candidate("checkpoint", root, task)
        manifest = _json(root / "model_index.json")
        required = {
            "transformer",
            "text_encoder",
            "video_vae",
            "audio_vae",
            "processor",
            "tokenizer",
        }
        if missing := required - manifest.keys():
            raise ValueError(
                "H3 task manifest lacks component declarations: " + ", ".join(sorted(missing))
            )
        _json(root / "audio_vae/metadata.json")
        inspect_safetensors_header(root / "audio_vae/model.safetensors")
        for key in ("audio_vae", "tokenizer", "processor"):
            _h3_candidate(key, root / key, task)


def _prepare_preconverted(record, files, cache, prepared):
    prepared.mkdir()
    for item in files:
        target = prepared / item.target
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link((cache / item.target).resolve(), target)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copy2(cache / item.target, target)
    _validate_preconverted(record["kind"], prepared)


def prepare_download(download_id, destination, *, progress=None, existing_roots=()):
    """Prepare a catalog component under a chosen library directory, without overwrites."""
    preconverted = next(
        (item for item in PRECONVERTED if item["descriptor"]["id"] == download_id), None
    )
    if not preconverted and download_id not in {"h3-qwen-q8-vision", "ltx25-distilled-q8"}:
        raise ValueError("Unknown model download; refresh the setup catalog")
    files = (
        tuple(DownloadFile(**item) for item in preconverted["files"])
        if preconverted
        else (QWEN_FILES if download_id == "h3-qwen-q8-vision" else LTX25_FILES)
    )
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / download_id
    if final.exists():
        raise FileExistsError(
            f"Model folder already exists: {final}. Use Existing Models to "
            "inspect it, or select another destination."
        )
    report = progress or (lambda *_: None)
    lock = destination / f".{download_id}.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError(
            f"Another setup may be using this destination. Lock: {lock}. "
            "After confirming no setup is running, remove the stale lock and retry."
        ) from None
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()))
        cache = destination / ".weetodd-downloads" / download_id
        cache.mkdir(parents=True, exist_ok=True)
        reuse_sources(files, cache, existing_roots, report)
        size = sum(item.size for item in files)
        remaining = sum(
            max(
                0,
                item.size
                - (
                    (cache / item.target).stat().st_size
                    if (cache / item.target).exists()
                    else partial_path(item, cache / item.target).stat().st_size
                    if partial_path(item, cache / item.target).exists()
                    else 0
                ),
            )
            for item in files
        )
        output_budget = size
        if preconverted:
            # New downloads and final files share hardlinks. Only reused sources on
            # another volume need an additional final copy on the destination volume.
            device = destination.stat().st_dev
            output_budget = sum(
                item.size
                for item in files
                if (cache / item.target).exists() and (cache / item.target).stat().st_dev != device
            )
        _check_space(destination, remaining + output_budget + RESERVE_BYTES)
        done = 0
        for item in files:
            target = cache / item.target
            target.parent.mkdir(parents=True, exist_ok=True)
            download_file(
                item,
                target,
                progress=lambda message, fraction, base=done, n=item.size: report(
                    message, 0.65 * (base + n * fraction) / size
                ),
            )
            done += item.size
        report(
            "Installing verified preconverted components…"
            if preconverted
            else "Preparing model pages; each page is checksum verified…",
            0.65,
        )
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=destination) as temporary:
            prepared = Path(temporary) / "model"
            if preconverted:
                _prepare_preconverted(preconverted, files, cache, prepared)
            elif download_id == "h3-qwen-q8-vision":
                _convert_qwen(cache / "compact", prepared, cache / "architecture/config.json")
            else:
                _convert_ltx25(cache, prepared, report)
            for name in ("LICENSE", "NOTICE", "MODIFICATIONS.md"):
                if (cache / "compact" / name).is_file():
                    shutil.copy2(cache / "compact" / name, prepared / name)
            (prepared / "setup_provenance.json").write_text(
                json.dumps(
                    {
                        "format": "weetodd-model-setup-v1",
                        **(
                            {
                                "engine": "h3",
                                "partition": preconverted["kind"].removeprefix("h3-transformer-"),
                            }
                            if preconverted and preconverted["kind"].startswith("h3-transformer-")
                            else {}
                        ),
                        "id": download_id,
                        "sources": [asdict(item) for item in files],
                        "converter": (
                            "preconverted"
                            if preconverted
                            else "weetodd-h3-qwen-paged-v2"
                            if download_id == "h3-qwen-q8-vision"
                            else "weetodd-ltx25-paged-q8-v1"
                        ),
                        "include_vision": (
                            preconverted["kind"] == "h3-qwen"
                            if preconverted
                            else download_id == "h3-qwen-q8-vision"
                        ),
                    },
                    indent=2,
                )
                + "\n"
            )
            if final.exists():
                raise FileExistsError(f"Model destination appeared during setup: {final}")
            publish_directory(prepared, final)
        report("Model preparation complete. Scan this folder to select its components.", 1.0)
        return {
            "path": str(final),
            "message": "Model components prepared and verified. "
            "Source downloads are retained in .weetodd-downloads for reuse."
            + (
                " H3 FL2VA/Ref2VA training identity cannot be inferred from shared tensor "
                "shapes; this package uses the pinned catalog source and checksums."
                if preconverted and preconverted["kind"].startswith("h3-transformer-")
                else ""
            ),
        }
    finally:
        lock.unlink(missing_ok=True)
