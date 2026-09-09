import json
from pathlib import Path

import pytest
from PIL import Image

from wee_todd_remote.adapter import DrawThingsAdapter
from wee_todd_remote.profiles import DrawThingsProfile


def request(profile_id="cloud"):
    return {
        "schema": "weetodd-drawthings-request-v1",
        "requestID": "request-1",
        "operation": "image",
        "profileID": profile_id,
        "modelID": "fixture-model",
        "prompt": "A paper bird",
        "negativePrompt": "",
        "configuration": {"width": 8, "height": 8, "steps": "automatic", "seed": 4},
        "inputs": [],
        "loras": [],
        "billingPolicy": "freeOnly",
    }


def profile(*, route="dtCloud", confirmed=False):
    return DrawThingsProfile(
        id="cloud",
        name="Fixture",
        route=route,
        host="example.invalid",
        port=443,
        useTLS=True,
        credentialRef="keychain-id",
        selfHostedConfirmed=confirmed,
    )


def discovery(*, authenticated=True):
    operation = {
        "width": {"min": 8, "max": 8, "multipleOf": 1},
        "height": {"min": 8, "max": 8, "multipleOf": 1},
        "inputRoleCombinations": [[]],
        "maxLoRAs": 0,
        "automaticSettings": {"steps": 4},
        "requiresAudio": False,
    }
    return {
        "authenticated": authenticated,
        "files": ["fixture-model"],
        "models": [{"id": "fixture-model", "name": "Fixture"}],
        "capabilities": {
            "fixture-model": {
                "operations": {"image": operation},
                "confidence": "verified",
                "source": "adapter-rules-intersect-endpoint-files",
                "revision": "r1",
            }
        },
        "transport": {"tls": True, "authenticated": authenticated},
    }


class HelperSpy:
    def __init__(self, *, discovery_value=None, estimate_configuration=None, manifest_path=None):
        self.calls = []
        self.discovery_value = discovery_value or discovery()
        self.estimate_configuration = estimate_configuration
        self.manifest_path = manifest_path

    def __call__(self, command, payload, *, helper, cancelled):
        self.calls.append((command, payload, helper, cancelled))
        if command == "capabilities":
            yield {
                "type": "result",
                "requestID": payload["requestID"],
                "value": self.discovery_value,
            }
        elif command == "estimate":
            configuration = self.estimate_configuration or payload["configuration"]
            yield {
                "type": "result",
                "requestID": payload["requestID"],
                "value": {
                    "cu": 2,
                    "estimatorRevision": "estimate-r1",
                    "configuration": configuration,
                },
            }
        else:
            yield {
                "type": "result",
                "requestID": payload["requestID"],
                "value": {"manifestPath": self.manifest_path},
            }


def adapter(spy, *, selected_profile=None, account_provider=None, credentials=None):
    selected_profile = selected_profile or profile()
    return DrawThingsAdapter(
        helper=Path("/fixture/helper"),
        profiles={selected_profile.id: selected_profile},
        credential_provider=lambda selected: credentials or {"apiKey": "private"},
        account_provider=account_provider,
        invoke=spy,
        now=lambda: 100,
    )


def free_account(_profile, _discovery):
    return {
        "authenticated": True,
        "limitMode": "cloud",
        "limitCU": 10,
        "policyExpiresAt": 200,
        "billingRoute": "free",
        "routeVerified": True,
    }


def test_prepare_uses_ephemeral_credentials_and_canonical_configuration():
    spy = HelperSpy()
    result = adapter(spy, account_provider=free_account).prepare(request())
    assert result["eligibility"] == "allowed"
    assert [call[0] for call in spy.calls] == ["capabilities", "estimate"]
    assert spy.calls[1][1]["configuration"]["steps"] == 4
    assert "request" not in spy.calls[1][1]
    assert spy.calls[0][1]["profile"]["credentialRef"] == "keychain-id"
    assert spy.calls[0][1]["credentials"] == {"apiKey": "private"}
    assert "credentials" not in result["normalizedRequest"]


def test_prepare_rejects_unknown_or_mismatched_profile_before_helper():
    spy = HelperSpy()
    selected = adapter(spy, account_provider=free_account)
    with pytest.raises(ValueError, match="profile"):
        selected.prepare(request("missing"))
    assert spy.calls == []


def test_generate_never_submits_when_cloud_policy_is_unknown(tmp_path):
    spy = HelperSpy()
    with pytest.raises(RuntimeError, match="unknown"):
        list(adapter(spy).generate(request(), tmp_path / "new", lambda: False))
    assert [call[0] for call in spy.calls] == ["capabilities", "estimate"]


def test_unsupported_model_blocks_before_estimate_or_generate(tmp_path):
    spy = HelperSpy()
    submitted = request()
    submitted["modelID"] = "not-discovered"
    with pytest.raises(RuntimeError, match="blocked"):
        list(
            adapter(spy, account_provider=free_account).generate(
                submitted, tmp_path / "new", lambda: False
            )
        )
    assert [call[0] for call in spy.calls] == ["capabilities"]


def test_lora_must_be_in_discovery_files_and_explicitly_compatible_before_estimate():
    submitted = request()
    submitted["loras"] = [{"modelID": "style-a", "weight": 0.75}]
    discovered = discovery()
    discovered["loras"] = [
        {
            "id": "style-a",
            "name": "Style A",
            "family": "ltx2.3",
            "compatibleModelIDs": ["fixture-model"],
        }
    ]
    spy = HelperSpy(discovery_value=discovered)
    with pytest.raises(ValueError, match="files"):
        adapter(spy, account_provider=free_account).prepare(submitted)
    assert [call[0] for call in spy.calls] == ["capabilities"]

    discovered["files"].append("style-a")
    discovered["loras"][0]["compatibleModelIDs"] = ["other-model"]
    spy = HelperSpy(discovery_value=discovered)
    with pytest.raises(ValueError, match="not compatible"):
        adapter(spy, account_provider=free_account).prepare(submitted)
    assert [call[0] for call in spy.calls] == ["capabilities"]


def test_verified_compatible_lora_reaches_estimate_unchanged():
    submitted = request()
    submitted["loras"] = [{"modelID": "style-a", "weight": 0.75}]
    discovered = discovery()
    discovered["files"].append("style-a")
    discovered["capabilities"]["fixture-model"]["operations"]["image"]["maxLoRAs"] = 1
    discovered["loras"] = [
        {
            "id": "style-a",
            "name": "Style A",
            "family": "ltx2.3",
            "compatibleModelIDs": ["fixture-model"],
        }
    ]
    spy = HelperSpy(discovery_value=discovered)
    prepared = adapter(spy, account_provider=free_account).prepare(submitted)
    assert prepared["eligibility"] == "allowed"
    assert spy.calls[1][1]["loras"] == [{"modelID": "style-a", "weight": 0.75}]


def test_adapter_rejects_stale_first_frame_hash_before_discovery(tmp_path):
    image = tmp_path / "first.png"
    image.write_bytes(b"current")
    submitted = request()
    submitted["operation"] = "video"
    submitted["inputs"] = [
        {"role": "first", "path": str(image), "sha256": "0" * 64, "frameIndex": 0, "strength": 1}
    ]
    spy = HelperSpy()
    with pytest.raises(ValueError, match="hash"):
        adapter(spy, account_provider=free_account).prepare(submitted)
    assert spy.calls == []


def test_generate_never_submits_stale_estimate_configuration(tmp_path):
    spy = HelperSpy(estimate_configuration={"width": 16, "height": 8, "steps": 4, "seed": 4})
    with pytest.raises(RuntimeError, match="unknown"):
        list(
            adapter(spy, account_provider=free_account).generate(
                request(), tmp_path / "new", lambda: False
            )
        )
    assert "generate" not in [call[0] for call in spy.calls]


def test_generate_creates_only_output_parent_before_helper(tmp_path):
    root = tmp_path / "render" / "media"
    spy = HelperSpy(manifest_path=str(root / "manifest.json"))
    iterator = adapter(spy, account_provider=free_account).generate(request(), root, lambda: False)
    with pytest.raises(ValueError):
        list(iterator)
    assert root.parent.is_dir()
    assert not root.exists()


def test_estimate_may_fill_defaults_and_returned_configuration_becomes_canonical():
    estimate_configuration = {
        "width": 8,
        "height": 8,
        "steps": 4,
        "seed": 4,
        "guidanceScale": 3.5,
        "strength": 1.0,
        "shift": 1.0,
        "sampler": 5,
    }
    spy = HelperSpy(estimate_configuration=estimate_configuration)
    prepared = adapter(spy, account_provider=free_account).prepare(request())
    assert prepared["eligibility"] == "allowed"
    assert prepared["normalizedRequest"]["configuration"] == estimate_configuration


def test_prepare_strips_imported_account_and_estimate_metadata():
    submitted = request()
    submitted["account"] = {"billingRoute": "paid"}
    submitted["estimate"] = {"cu": 1}
    prepared = adapter(HelperSpy(), account_provider=free_account).prepare(submitted)
    assert "account" not in prepared["normalizedRequest"]
    assert "estimate" not in prepared["normalizedRequest"]


def test_confirmed_authenticated_grpc_is_exempt_without_cloud_account(tmp_path):
    local = profile(route="grpc", confirmed=True)
    spy = HelperSpy()
    prepared = adapter(spy, selected_profile=local).prepare(request())
    assert prepared["eligibility"] == "allowed"
    assert prepared["limitMode"] == "notApplicable"


def write_image_manifest(root, *, request_id="request-1", image_path="00000000.png"):
    root.mkdir()
    Image.new("RGB", (8, 8), "red").save(root / "00000000.png")
    manifest = {
        "schema": "weetodd-drawthings-media-v1",
        "requestID": request_id,
        "operation": "image",
        "frameCount": 1,
        "configuration": {"width": 8, "height": 8},
        "imagePaths": [image_path],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_generate_validates_real_image_manifest_and_reuses_canonical_request(tmp_path):
    output = tmp_path / "output"
    write_image_manifest(output)
    # Adapter requires ownership of a new directory, so stage helper artifacts during generation.
    output.rename(tmp_path / "staged")
    spy = HelperSpy(manifest_path=str(output / "manifest.json"))

    original_call = spy.__call__

    def invoke(command, payload, **kwargs):
        if command == "generate":
            (tmp_path / "staged").rename(output)
        yield from original_call(command, payload, **kwargs)

    selected = DrawThingsAdapter(
        helper=Path("/fixture/helper"),
        profiles={"cloud": profile()},
        credential_provider=lambda _: {"apiKey": "private"},
        account_provider=free_account,
        invoke=invoke,
        now=lambda: 100,
    )
    events = list(selected.generate(request(), output, lambda: False))
    assert events[-1]["value"]["media"]["imagePaths"] == [str(output / "00000000.png")]
    assert events[-1]["value"]["fingerprint"]
    assert events[-1]["value"]["normalizedRequest"]["configuration"]["steps"] == 4
    for key in request():
        assert spy.calls[1][1][key] == spy.calls[2][1][key]
    assert "request" not in spy.calls[2][1]


@pytest.mark.parametrize("manifest_path", ["../manifest.json", "/tmp/manifest.json"])
def test_generate_rejects_manifest_path_escape(tmp_path, manifest_path):
    spy = HelperSpy(manifest_path=manifest_path)
    with pytest.raises(ValueError, match="manifest"):
        list(
            adapter(spy, account_provider=free_account).generate(
                request(), tmp_path / "output", lambda: False
            )
        )


def test_generate_rejects_mismatched_manifest_request_id(tmp_path):
    output = tmp_path / "output"
    spy = HelperSpy(manifest_path=str(output / "manifest.json"))
    original_call = spy.__call__

    def invoke(command, payload, **kwargs):
        if command == "generate":
            write_image_manifest(output, request_id="other")
        yield from original_call(command, payload, **kwargs)

    selected = adapter(spy, account_provider=free_account)
    selected._invoke = invoke
    with pytest.raises(ValueError, match="requestID"):
        list(selected.generate(request(), output, lambda: False))


def test_generate_propagates_cancellation_callback(tmp_path):
    spy = HelperSpy(manifest_path="manifest.json")

    def cancelled():
        return False

    original_call = spy.__call__

    def invoke(command, payload, *, helper, cancelled):
        assert cancelled is cancellation_callback
        if command == "generate":
            raise InterruptedError("Draw Things request cancelled")
        yield from original_call(command, payload, helper=helper, cancelled=cancelled)

    cancellation_callback = cancelled
    selected = DrawThingsAdapter(
        helper=Path("/fixture/helper"),
        profiles={"cloud": profile()},
        credential_provider=lambda _: {"apiKey": "private"},
        account_provider=free_account,
        invoke=invoke,
        now=lambda: 100,
    )
    with pytest.raises(InterruptedError, match="cancel"):
        list(selected.generate(request(), tmp_path / "output", cancellation_callback))


def test_generate_yields_progress_before_helper_result_and_validation(tmp_path):
    output = tmp_path / "output"
    spy = HelperSpy(manifest_path=str(output / "manifest.json"))
    original_call = spy.__call__

    def invoke(command, payload, **kwargs):
        if command != "generate":
            yield from original_call(command, payload, **kwargs)
            return
        yield {"type": "progress", "requestID": "request-1", "value": {"fraction": 0.5}}
        write_image_manifest(output)
        yield {
            "type": "result",
            "requestID": "request-1",
            "value": {"manifestPath": str(output / "manifest.json")},
        }

    selected = DrawThingsAdapter(
        helper=Path("/fixture/helper"),
        profiles={"cloud": profile()},
        credential_provider=lambda _: {"apiKey": "private"},
        account_provider=free_account,
        invoke=invoke,
        now=lambda: 100,
    )
    events = selected.generate(request(), output, lambda: False)
    assert next(events)["type"] == "progress"
    assert next(events)["type"] == "result"


@pytest.mark.parametrize("inventory", [[], ["different-model"], None])
def test_capabilities_cannot_add_models_absent_from_endpoint_files(inventory):
    value = discovery()
    value["files"] = inventory
    spy = HelperSpy(discovery_value=value)
    result = adapter(spy, account_provider=free_account).prepare(request())
    assert result["eligibility"] == "blocked"
    assert [call[0] for call in spy.calls] == ["capabilities"]


def test_self_hosted_keeps_estimate_visible_without_a_cloud_limit():
    spy = HelperSpy()
    result = adapter(spy, selected_profile=profile(route="grpc", confirmed=True)).prepare(request())
    assert result["eligibility"] == "allowed"
    assert result["limitMode"] == "notApplicable"
    assert result["limitCU"] is None
    assert result["estimateCU"] == 2
    assert result["estimateSource"] == "estimate-r1"
