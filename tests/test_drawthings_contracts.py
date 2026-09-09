import pytest

from wee_todd_remote.contracts import (
    request_fingerprint,
    validate_event,
    validate_request,
)
from wee_todd_remote.profiles import DrawThingsProfile


@pytest.fixture
def image_request():
    return {
        "schema": "weetodd-drawthings-request-v1",
        "requestID": "fixture-request-1",
        "operation": "image",
        "profileID": "drawthings-local",
        "modelID": "fixture-image-model",
        "prompt": "A red cube on a gray surface",
        "negativePrompt": "",
        "configuration": {"width": 512, "height": 512, "steps": 4, "seed": 42},
        "inputs": [],
        "loras": [],
        "billingPolicy": "freeOnly",
    }


def test_request_validation_normalizes_optional_collections(image_request):
    value = {k: v for k, v in image_request.items() if k not in {"inputs", "loras"}}
    assert validate_request(value) == image_request


@pytest.mark.parametrize("field", ["requestID", "profileID", "modelID", "prompt"])
def test_request_rejects_missing_or_empty_text_fields(image_request, field):
    image_request[field] = ""
    with pytest.raises(ValueError, match=field):
        validate_request(image_request)


def test_request_rejects_unknown_operation_and_billing_policy(image_request):
    image_request["operation"] = "audio"
    with pytest.raises(ValueError, match="operation"):
        validate_request(image_request)
    image_request["operation"] = "video"
    image_request["billingPolicy"] = "payAsYouGo"
    with pytest.raises(ValueError, match="billingPolicy"):
        validate_request(image_request)


def test_request_rejects_non_json_or_invalid_container_values(image_request):
    image_request["configuration"] = {"bad": object()}
    with pytest.raises(ValueError, match="configuration"):
        validate_request(image_request)
    image_request["configuration"] = {}
    image_request["inputs"] = {}
    with pytest.raises(ValueError, match="inputs"):
        validate_request(image_request)


def test_cu_refresh_does_not_change_generation_identity(image_request):
    refreshed = {**image_request, "estimate": {"cu": 123, "expiresAt": 200}}
    assert request_fingerprint(refreshed) == request_fingerprint(image_request)


def test_request_id_and_account_do_not_change_generation_identity(image_request):
    changed = {**image_request, "requestID": "retry-request-2", "account": {"remainingCU": 1}}
    assert request_fingerprint(changed) == request_fingerprint(image_request)


def test_credentials_are_prohibited_from_portable_requests(image_request):
    image_request["credentials"] = {"token": "secret"}
    with pytest.raises(ValueError, match="credentials"):
        validate_request(image_request)


@pytest.mark.parametrize(
    "secret_key",
    [
        "apiKey",
        "token",
        "authorization",
        "sharedSecret",
        "password",
        "client_secret",
        "access-token",
        "refreshToken",
        "privateKey",
        "authToken",
        "bearer_token",
        "secret-key",
    ],
)
def test_nested_secret_bearing_keys_are_prohibited(image_request, secret_key):
    image_request["configuration"] = {"transport": {secret_key: "secret"}}
    with pytest.raises(ValueError, match="secret"):
        validate_request(image_request)


def test_benign_token_configuration_names_are_allowed(image_request):
    image_request["configuration"] = {"tokenizer": "t5", "tokenCount": 128}
    assert validate_request(image_request)["configuration"] == {
        "tokenizer": "t5",
        "tokenCount": 128,
    }


def test_request_rejects_credential_reference_and_unknown_top_level_fields(image_request):
    image_request["credentialRef"] = "drawthings-keychain"
    with pytest.raises(ValueError, match="credentialRef"):
        validate_request(image_request)
    image_request.pop("credentialRef")
    image_request["futureExtension"] = True
    with pytest.raises(ValueError, match="futureExtension"):
        validate_request(image_request)


@pytest.mark.parametrize("field", ["operation", "billingPolicy"])
def test_unhashable_enum_values_raise_value_error(image_request, field):
    image_request[field] = []
    with pytest.raises(ValueError, match=field):
        validate_request(image_request)


@pytest.mark.parametrize("bad_value", [{1: "value"}, ("tuple",)])
def test_python_only_json_coercions_are_rejected(image_request, bad_value):
    image_request["configuration"] = {"bad": bad_value}
    with pytest.raises(ValueError, match="configuration"):
        validate_request(image_request)


def test_remote_model_changes_generation_identity(image_request):
    changed = {**image_request, "modelID": "another-fixture-model"}
    assert request_fingerprint(changed) != request_fingerprint(image_request)


@pytest.mark.parametrize("event_type", ["progress", "preview", "result", "error"])
def test_event_validation_accepts_supported_messages(event_type):
    event = {"requestID": "fixture-request-1", "type": event_type, "detail": {"value": 1}}
    assert validate_event(event) == event


def test_event_validation_rejects_bad_envelopes():
    with pytest.raises(ValueError, match="requestID"):
        validate_event({"requestID": "", "type": "progress"})
    with pytest.raises(ValueError, match="type"):
        validate_event({"requestID": "fixture-request-1", "type": "unknown"})


def test_profile_keeps_connection_route_separate_from_model_selection():
    profile = DrawThingsProfile(
        id="drawthings-local",
        name="Local Draw Things",
        route="grpc",
        host="127.0.0.1",
        port=7859,
        useTLS=False,
        credentialRef="drawthings-keychain",
    )
    assert profile.id == "drawthings-local"
    assert profile.to_dict()["credentialRef"] == "drawthings-keychain"
    assert not hasattr(profile, "modelID")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"route": "shell"}, "route"),
        ({"route": []}, "route"),
        ({"host": ""}, "host"),
        ({"host": "user:password@example.invalid"}, "host"),
        ({"host": "https://user@example.invalid"}, "host"),
        ({"host": "user%40example.invalid"}, "host"),
        ({"port": 0}, "port"),
        ({"port": True}, "port"),
        ({"useTLS": "yes"}, "useTLS"),
    ],
)
def test_profile_rejects_invalid_connection_fields(change, message):
    values = {
        "id": "drawthings-local",
        "name": "Local Draw Things",
        "route": "grpc",
        "host": "127.0.0.1",
        "port": 7859,
        "useTLS": False,
    }
    with pytest.raises(ValueError, match=message):
        DrawThingsProfile(**(values | change))


def test_profile_serialization_contains_reference_but_no_secret():
    profile = DrawThingsProfile(
        id="cloud",
        name="Cloud",
        route="dtCloud",
        host="api.example.invalid",
        port=443,
        useTLS=True,
        credentialRef="keychain-item",
    )
    serialized = profile.to_dict()
    assert serialized == {
        "id": "cloud",
        "name": "Cloud",
        "route": "dtCloud",
        "host": "api.example.invalid",
        "port": 443,
        "useTLS": True,
        "credentialRef": "keychain-item",
        "selfHostedConfirmed": False,
    }


def test_self_hosted_confirmation_is_a_local_profile_setting():
    values = dict(id="local", name="Local", route="grpc", host="localhost", port=7859, useTLS=False)
    assert DrawThingsProfile(**values, selfHostedConfirmed=True).selfHostedConfirmed
    for bad in ("yes", 1, None):
        with pytest.raises(ValueError):
            DrawThingsProfile(**values, selfHostedConfirmed=bad)
    with pytest.raises(ValueError):
        DrawThingsProfile(**(values | {"route": "dtBridge"}), selfHostedConfirmed=True)
