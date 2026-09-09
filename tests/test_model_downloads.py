"""Managed preparation never installs unverified or partial components."""

import hashlib
import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_preflight import _component_tree, _json, _portable_optimized_spec, _safetensors

from wee_todd_mlx import model_downloads as downloads


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}


def spec(body=b"checkpoint"):
    return downloads.DownloadFile(
        repo="owner/model",
        revision="a" * 40,
        filename="model.safetensors",
        size=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
        target="model.safetensors",
    )


def test_download_verifies_then_reuses_without_network(tmp_path):
    item = spec()
    opener = Mock(return_value=Response(b"checkpoint"))
    target = tmp_path / "model.safetensors"
    downloads.download_file(item, target, opener=opener)
    assert target.read_bytes() == b"checkpoint"
    downloads.download_file(item, target, opener=opener)
    assert opener.call_count == 1


def test_resume_uses_range_and_checks_content_range(tmp_path):
    item = spec()
    target = tmp_path / "model.safetensors"
    downloads.partial_path(item, target).write_bytes(b"check")
    opener = Mock(return_value=Response(b"point", 206, {"Content-Range": "bytes 5-9/10"}))
    downloads.download_file(item, target, opener=opener)
    assert opener.call_args.args[0].get_header("Range") == "bytes=5-"
    assert target.read_bytes() == b"checkpoint"


def test_server_ignoring_range_restarts_safely(tmp_path):
    item = spec()
    target = tmp_path / "model.safetensors"
    downloads.partial_path(item, target).write_bytes(b"check")
    downloads.download_file(item, target, opener=lambda *a, **kw: Response(b"checkpoint"))
    assert target.read_bytes() == b"checkpoint"


def test_mismatched_range_preserves_partial(tmp_path):
    item = spec()
    target = tmp_path / "model.safetensors"
    part = downloads.partial_path(item, target)
    part.write_bytes(b"check")
    with pytest.raises(ValueError, match="range"):
        downloads.download_file(
            item,
            target,
            opener=lambda *a, **kw: Response(b"point", 206, {"Content-Range": "bytes 0-4/10"}),
        )
    assert part.read_bytes() == b"check"
    assert not target.exists()


def test_hash_mismatch_is_never_published(tmp_path):
    item = spec()
    target = tmp_path / "model.safetensors"
    with pytest.raises(ValueError, match="SHA-256"):
        downloads.download_file(item, target, opener=lambda *a, **kw: Response(b"XXXXXXXXXX"))
    assert not target.exists()
    assert not downloads.partial_path(item, target).exists()


def test_short_response_retains_partial_for_retry(tmp_path):
    item = spec()
    target = tmp_path / "model.safetensors"
    with pytest.raises(ValueError, match="incomplete"):
        downloads.download_file(item, target, opener=lambda *a, **kw: Response(b"check"))
    assert downloads.partial_path(item, target).read_bytes() == b"check"
    assert not target.exists()


def test_oversized_response_never_replaces_existing_file(tmp_path):
    item = spec()
    target = tmp_path / "model.safetensors"
    target.write_bytes(b"existing user file")
    with pytest.raises(ValueError, match="existing"):
        downloads.download_file(item, target, opener=lambda *a, **kw: Response(b"checkpoint"))
    assert target.read_bytes() == b"existing user file"


def test_catalog_is_offline_pinned_and_has_space_estimates():
    catalog = downloads.download_catalog()
    assert any(item["id"] == "h3-qwen-q8-vision" for item in catalog)
    for item in catalog:
        assert item["requiredDiskBytes"] >= item["downloadBytes"] > 0
        assert item["licenseURL"].startswith("https://")
    for item in downloads.QWEN_FILES:
        assert len(item.revision) == 40 and len(item.sha256) == 64


def test_prepare_failure_leaves_working_models_and_no_success_output(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "QWEN_FILES", (spec(),))
    monkeypatch.setattr(downloads, "_check_space", lambda *args: None)
    monkeypatch.setattr(
        downloads, "download_file", lambda item, target, **kw: target.write_bytes(b"checkpoint")
    )
    monkeypatch.setattr(downloads, "_convert_qwen", Mock(side_effect=ValueError("bad config")))
    old = tmp_path / "existing-model"
    old.mkdir()
    (old / "weights").write_bytes(b"original")
    with pytest.raises(ValueError, match="bad config"):
        downloads.prepare_download("h3-qwen-q8-vision", tmp_path)
    assert not (tmp_path / "h3-qwen-q8-vision").exists()
    assert (old / "weights").read_bytes() == b"original"
    assert not list(tmp_path.glob(".prepare-*"))
    assert not list(tmp_path.glob("*.lock"))


def test_prepare_success_records_provenance_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "QWEN_FILES", (spec(),))
    monkeypatch.setattr(downloads, "_check_space", lambda *args: None)
    monkeypatch.setattr(
        downloads, "download_file", lambda item, target, **kw: target.write_bytes(b"checkpoint")
    )

    def convert(source, destination, architecture):
        destination.mkdir()
        (destination / "paged_text_encoder_manifest.json").write_text("{}")

    monkeypatch.setattr(downloads, "_convert_qwen", convert)
    result = downloads.prepare_download("h3-qwen-q8-vision", tmp_path)
    assert result["path"] == str(tmp_path / "h3-qwen-q8-vision")
    assert (
        json.loads((tmp_path / "h3-qwen-q8-vision" / "setup_provenance.json").read_text())[
            "sources"
        ][0]["sha256"]
        == spec().sha256
    )
    with pytest.raises(FileExistsError):
        downloads.prepare_download("h3-qwen-q8-vision", tmp_path)


def test_github_license_source_has_no_huggingface_token(tmp_path, monkeypatch):
    item = downloads.DownloadFile(
        "owner/model",
        "a" * 40,
        "LICENSE",
        4,
        hashlib.sha256(b"text").hexdigest(),
        "LICENSE",
        provider="github",
    )
    monkeypatch.setattr(downloads, "_hf_token", lambda: "private-token")
    opener = Mock(return_value=Response(b"text"))
    downloads.download_file(item, tmp_path / "LICENSE", opener=opener)
    request = opener.call_args.args[0]
    assert request.full_url.startswith("https://raw.githubusercontent.com/")
    assert request.get_header("Authorization") is None


def test_ltx25_download_catalog_covers_complete_distilled_component_set():
    item = next(item for item in downloads.download_catalog() if item["id"] == "ltx25-distilled-q8")
    assert item["engines"] == ["ltx25"]
    assert len(downloads.LTX25_FILES) >= 6  # five components and source terms
    assert any("conv-bf16" in item.filename for item in downloads.LTX25_FILES)


def test_ltx25_bundle_preparation_is_staged(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "LTX25_FILES", (spec(),))
    monkeypatch.setattr(downloads, "_check_space", lambda *args: None)
    monkeypatch.setattr(
        downloads, "download_file", lambda item, target, **kw: target.write_bytes(b"checkpoint")
    )
    converter = Mock(side_effect=lambda source, destination, progress: destination.mkdir())
    monkeypatch.setattr(downloads, "_convert_ltx25", converter)
    result = downloads.prepare_download("ltx25-distilled-q8", tmp_path)
    assert result["path"] == str(tmp_path / "ltx25-distilled-q8")
    assert converter.call_count == 1
    provenance = json.loads((tmp_path / "ltx25-distilled-q8/setup_provenance.json").read_text())
    assert provenance["converter"] == "weetodd-ltx25-paged-q8-v1"
    assert provenance["include_vision"] is False
    assert not list(tmp_path.glob(".prepare-*"))


def test_existing_sources_are_reused_only_after_full_hash_match(tmp_path):
    item = spec()
    source = tmp_path / "source"
    source.mkdir()
    original = source / "renamed-weight.safetensors"
    original.write_bytes(b"checkpoint")
    (source / "cycle").symlink_to(source, target_is_directory=True)
    cache = tmp_path / "cache"
    cache.mkdir()
    downloads.reuse_sources((item,), cache, [str(source)])
    cached = cache / item.target
    assert cached.read_bytes() == b"checkpoint"
    assert cached.stat().st_ino == original.stat().st_ino


def test_same_size_wrong_source_is_not_reused(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"XXXXXXXXXX")
    cache = tmp_path / "cache"
    cache.mkdir()
    downloads.reuse_sources((spec(),), cache, [str(source)])
    assert not (cache / spec().target).exists()


def test_directory_publish_preserves_competing_empty_directory(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "new").write_text("new")
    target = tmp_path / "target"
    target.mkdir()
    original_inode = target.stat().st_ino
    with pytest.raises(FileExistsError):
        downloads.publish_directory(source, target)
    assert target.stat().st_ino == original_inode
    assert (source / "new").exists()


def test_redirect_strips_token_for_different_origin():
    import urllib.request

    handler = downloads._SafeRedirect()
    request = urllib.request.Request(
        "https://huggingface.co/file", headers={"Authorization": "Bearer secret"}
    )
    redirected = handler.redirect_request(
        request, None, 302, "", {}, "https://huggingface.co:444/file"
    )
    assert redirected.get_header("Authorization") is None
    with pytest.raises(ValueError, match="insecure"):
        handler.redirect_request(request, None, 302, "", {}, "http://huggingface.co/file")


def test_cancellation_keeps_partial_and_releases_setup_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(downloads, "QWEN_FILES", (spec(),))
    monkeypatch.setattr(downloads, "_check_space", lambda *args: None)

    def interrupted(item, target, **kwargs):
        downloads.partial_path(item, target).write_bytes(b"check")
        raise KeyboardInterrupt

    monkeypatch.setattr(downloads, "download_file", interrupted)
    with pytest.raises(KeyboardInterrupt):
        downloads.prepare_download("h3-qwen-q8-vision", tmp_path)
    assert not list(tmp_path.glob("*.lock"))
    assert not (tmp_path / "h3-qwen-q8-vision").exists()
    assert list(tmp_path.rglob("*.part"))


def test_insufficient_disk_space_fails_before_downloading(tmp_path, monkeypatch):
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(downloads.shutil, "disk_usage", lambda _: usage(10, 10, 0))
    download = Mock()
    monkeypatch.setattr(downloads, "download_file", download)
    with pytest.raises(ValueError, match="free disk space"):
        downloads.prepare_download("h3-qwen-q8-vision", tmp_path)
    download.assert_not_called()
    assert not list(tmp_path.glob("*.lock"))


def test_preconverted_download_skips_quantization(tmp_path, monkeypatch):
    descriptor = {
        "id": "fixture-preconverted",
        "name": "Fixture",
        "description": "Fixture",
        "downloadBytes": 10,
        "requiredDiskBytes": 10,
        "sourceURL": "https://huggingface.co/owner/model",
        "licenseURL": "https://huggingface.co/owner/model",
        "outputKind": "directory",
        "engines": ["h3"],
    }
    record = {"descriptor": descriptor, "kind": "h3-qwen", "files": [downloads.asdict(spec())]}
    monkeypatch.setattr(downloads, "PRECONVERTED", [record])
    space_check = Mock()
    monkeypatch.setattr(downloads, "_check_space", space_check)
    monkeypatch.setattr(
        downloads, "download_file", lambda item, target, **kw: target.write_bytes(b"checkpoint")
    )
    monkeypatch.setattr(downloads, "_validate_preconverted", lambda *a: None)
    convert = Mock(side_effect=AssertionError("must not quantize preconverted models"))
    monkeypatch.setattr(downloads, "_convert_qwen", convert)
    result = downloads.prepare_download("fixture-preconverted", tmp_path)
    assert (tmp_path / "fixture-preconverted/model.safetensors").read_bytes() == b"checkpoint"
    assert result["path"] == str(tmp_path / "fixture-preconverted")
    assert downloads.download_catalog()[0]["id"] == "fixture-preconverted"
    provenance = json.loads((tmp_path / "fixture-preconverted/setup_provenance.json").read_text())
    assert provenance["converter"] == "preconverted"
    convert.assert_not_called()
    assert space_check.call_args.args[1] == 10 + downloads.RESERVE_BYTES


def test_oversized_stream_remains_unpublished(tmp_path):
    item = spec()
    target = tmp_path / "model.safetensors"
    with pytest.raises(ValueError, match="exceeds"):
        downloads.download_file(item, target, opener=lambda *a, **kw: Response(b"checkpointEXTRA"))
    assert not target.exists()
    assert downloads.partial_path(item, target).stat().st_size <= item.size


def test_ltx_conversion_copies_cross_volume_reused_support_files(tmp_path, monkeypatch):
    import errno

    import ltx25_mlx.paged_checkpoint as paged
    import ltx25_mlx.runtime as runtime

    source = tmp_path / "source"
    source.mkdir()
    for item in downloads.LTX25_FILES:
        filename = source / item.target
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_bytes(b"source")
    converted = []

    def convert(source, destination, **kwargs):
        converted.append(kwargs["kind"])
        destination.mkdir()

    monkeypatch.setattr(paged, "convert_to_paged_q8", convert)
    monkeypatch.setattr(runtime, "LTX25ComponentSpec", Mock())
    monkeypatch.setattr(
        downloads.os, "link", Mock(side_effect=OSError(errno.EXDEV, "cross-volume"))
    )
    output = tmp_path / "prepared"
    downloads._convert_ltx25(source, output, lambda *a: None)
    assert converted == ["transformer", "gemma"]
    assert (output / "vae/ltx-2.5-video-vae-conv-bf16.safetensors").read_bytes() == b"source"
    assert (output / "LICENSE-2_x").read_bytes() == b"source"


def test_shipped_preconverted_catalog_pins_files_and_retains_terms():
    assert downloads.PRECONVERTED
    for package in downloads.PRECONVERTED:
        items = [downloads.DownloadFile(**item) for item in package["files"]]
        assert len({item.revision for item in items}) == 1
        assert len({item.target for item in items}) == len(items)
        assert sum(item.size for item in items) == package["descriptor"]["downloadBytes"]
        assert all(
            item.repo.startswith("Vayden/") or item.repo == "MiniMaxAI/MiniMax-H3" for item in items
        )
        assert any("LICENSE" in item.target for item in items)
        if package["kind"].startswith("h3-support-"):
            assert {
                "model_index.json",
                "audio_vae/config.json",
                "audio_vae/metadata.json",
                "audio_vae/model.safetensors",
            } <= {item.target for item in items}
        elif package["kind"] == "h3-video-vae":
            assert "video_vae_affine_q8.safetensors" in {item.target for item in items}
        else:
            assert any(item.target.endswith("manifest.json") for item in items)
        assert "Recommended" in package["descriptor"]["name"]


def test_h3_and_ltx25_presets_have_downloads_for_every_required_component():
    from wee_todd_mlx.model_setup import setup_catalog

    for preset in setup_catalog():
        if preset["engine"] not in {"h3", "ltx25"}:
            continue
        provided = {
            component
            for download in downloads.download_catalog()
            if preset["engine"] in download.get("engines", [])
            and preset["task"] in download.get("tasks", [preset["task"]])
            for component in download.get("components", [])
        }
        required = {component["key"] for component in preset["components"]}
        assert required <= provided, (preset["id"], required - provided)


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_preconverted_h3_transformer_accepts_valid_pages_without_inventing_identity(tmp_path, task):
    spec = _portable_optimized_spec(tmp_path)
    downloads._validate_preconverted(f"h3-transformer-{task}", Path(spec.transformer))


@pytest.mark.parametrize("defect", ["missing-page", "missing-manifest", "block-count", "config"])
def test_preconverted_h3_transformer_rejects_incomplete_package(tmp_path, defect):
    root = Path(_portable_optimized_spec(tmp_path).transformer)
    if defect == "missing-page":
        (root / "pages/block-000.safetensors").unlink()
    elif defect == "missing-manifest":
        (root / "paged_manifest.json").unlink()
    elif defect == "block-count":
        manifest = json.loads((root / "paged_manifest.json").read_text())
        manifest["num_blocks"] = 2
        _json(root / "paged_manifest.json", manifest)
    else:
        _json(root / "config.json", {"latents_dim": 128})
    with pytest.raises((ValueError, FileNotFoundError), match="page|manifest|blocks|architecture"):
        downloads._validate_preconverted("h3-transformer-fl2va", root)


@pytest.mark.parametrize("task,other", [("fl2va", "ref2va"), ("ref2va", "fl2va")])
@pytest.mark.parametrize(
    "filename",
    ["config.json", "paged_manifest.json", "conversion_provenance.json", "model_identity.json"],
)
def test_preconverted_transformer_rejects_explicit_wrong_partition(tmp_path, task, other, filename):
    root = Path(_portable_optimized_spec(tmp_path).transformer)
    target = root / filename
    manifest = json.loads(target.read_text()) if target.exists() else {}
    if filename == "model_identity.json":
        manifest["partition"] = other
    else:
        manifest["_minimax_h3"] = {"partition": other, "tasks": [other]}
    _json(target, manifest)
    with pytest.raises(ValueError, match="partition"):
        downloads._validate_preconverted(f"h3-transformer-{task}", root)


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_preconverted_h3_support_validates_task_and_native_components(tmp_path, task):
    root = _component_tree(tmp_path, tasks=(task,))
    manifest = json.loads((root / "model_index.json").read_text())
    manifest["_minimax_h3"]["partition"] = task
    _json(root / "model_index.json", manifest)
    downloads._validate_preconverted(f"h3-support-{task}", root)


@pytest.mark.parametrize(
    "missing",
    [
        "model_index.json",
        "audio_vae/model.safetensors",
        "audio_vae/metadata.json",
        "tokenizer/tokenizer.json",
        "processor/preprocessor_config.json",
    ],
)
def test_preconverted_h3_support_rejects_missing_component(tmp_path, missing):
    root = _component_tree(tmp_path)
    (root / missing).unlink()
    with pytest.raises(FileNotFoundError):
        downloads._validate_preconverted("h3-support-fl2va", root)


def test_preconverted_h3_support_rejects_fl2va_as_reference(tmp_path):
    root = _component_tree(tmp_path)
    with pytest.raises(ValueError, match="task|partition"):
        downloads._validate_preconverted("h3-support-ref2va", root)


def test_preconverted_h3_video_vae_checks_self_describing_header(tmp_path):
    filename = Path(_portable_optimized_spec(tmp_path).video_vae)
    downloads._validate_preconverted("h3-video-vae", filename.parent)


@pytest.mark.parametrize("defect", ["wrong-header", "missing-metadata", "wrong-wrapper"])
def test_preconverted_h3_video_vae_rejects_invalid_weights(tmp_path, defect):
    filename = tmp_path / "video_vae_affine_q8.safetensors"
    if defect == "wrong-header":
        filename.write_bytes(b"not a checkpoint")
    else:
        metadata = {} if defect == "missing-metadata" else {"minimax_h3_video_vae": "{}"}
        _safetensors(filename, {"decoder.weight": ("F16", [4, 4], 32)}, metadata)
    with pytest.raises(ValueError, match="header|metadata|format"):
        downloads._validate_preconverted("h3-video-vae", tmp_path)


@pytest.mark.parametrize("valid", [True, False])
def test_prepare_h3_transformer_checks_before_publish_and_reports_identity_limit(
    tmp_path, monkeypatch, valid
):
    source = Path(_portable_optimized_spec(tmp_path).transformer)
    if not valid:
        (source / "pages/block-000.safetensors").unlink()
    payloads = {
        str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    items = [
        downloads.DownloadFile(
            "owner/model", "a" * 40, name, len(body), hashlib.sha256(body).hexdigest(), name
        )
        for name, body in payloads.items()
    ]
    record = {
        "descriptor": {"id": "fixture-transformer"},
        "kind": "h3-transformer-fl2va",
        "files": [downloads.asdict(item) for item in items],
    }
    monkeypatch.setattr(downloads, "PRECONVERTED", [record])
    monkeypatch.setattr(
        downloads,
        "_open",
        lambda request, **kw: Response(
            payloads[request.full_url.split("/resolve/" + "a" * 40 + "/")[1]]
        ),
    )
    monkeypatch.setattr(downloads, "_hf_token", lambda: None)
    destination = tmp_path / "downloaded"
    if valid:
        result = downloads.prepare_download("fixture-transformer", destination)
        assert "training identity" in result["message"]
        assert "pinned" in result["message"]
        assert (destination / "fixture-transformer/paged_manifest.json").is_file()
        provenance = json.loads(
            (destination / "fixture-transformer/setup_provenance.json").read_text()
        )
        assert provenance["engine"] == "h3"
        assert provenance["partition"] == "fl2va"
    else:
        with pytest.raises(FileNotFoundError, match="page"):
            downloads.prepare_download("fixture-transformer", destination)
        assert not (destination / "fixture-transformer").exists()
        assert not list(destination.glob(".prepare-*"))
