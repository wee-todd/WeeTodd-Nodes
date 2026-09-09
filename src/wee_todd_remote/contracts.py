from __future__ import annotations

import hashlib
import json
import math
from typing import Any

REQUEST_SCHEMA = "weetodd-drawthings-request-v1"
OPERATIONS = frozenset({"image", "video"})
BILLING_POLICIES = frozenset({"freeOnly"})
EVENT_TYPES = frozenset({"progress", "preview", "result", "error"})
_TRANSIENT_REQUEST_KEYS = frozenset({"requestID", "estimate", "account"})
_SECRET_KEYS = frozenset(
    {
        "credentials",
        "apikey",
        "token",
        "authorization",
        "sharedsecret",
        "password",
        "clientsecret",
        "accesstoken",
        "refreshtoken",
        "privatekey",
        "authtoken",
        "bearertoken",
        "secretkey",
    }
)
_REQUEST_KEYS = frozenset(
    {
        "schema",
        "requestID",
        "operation",
        "profileID",
        "modelID",
        "prompt",
        "negativePrompt",
        "configuration",
        "inputs",
        "loras",
        "billingPolicy",
        "estimate",
        "account",
    }
)


def _nonempty_string(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return result


def _json_value(value: Any, key: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        raise ValueError(f"{key} must contain finite JSON numbers")
    if isinstance(value, list):
        for item in value:
            _json_value(item, key)
        return value
    if isinstance(value, dict):
        for item_key, item in value.items():
            if not isinstance(item_key, str):
                raise ValueError(f"{key} must contain string object keys")
            normalized_key = "".join(
                character for character in item_key.lower() if character.isalnum()
            )
            if normalized_key in _SECRET_KEYS:
                raise ValueError(f"{key} contains prohibited secret-bearing field {item_key}")
            _json_value(item, key)
        return value
    raise ValueError(f"{key} must contain JSON values")
    return value


def validate_request(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("request must be an object")
    normalized = dict(value)
    unexpected = normalized.keys() - _REQUEST_KEYS
    if unexpected:
        field = min(str(item) for item in unexpected)
        raise ValueError(f"unsupported request field {field}")
    if normalized.get("schema") != REQUEST_SCHEMA:
        raise ValueError(f"schema must be {REQUEST_SCHEMA}")
    for key in ("requestID", "profileID", "modelID", "prompt"):
        _nonempty_string(normalized, key)
    operation = normalized.get("operation")
    if not isinstance(operation, str) or operation not in OPERATIONS:
        raise ValueError("operation must be image or video")
    billing_policy = normalized.get("billingPolicy")
    if not isinstance(billing_policy, str) or billing_policy not in BILLING_POLICIES:
        raise ValueError("billingPolicy must be freeOnly")
    negative_prompt = normalized.get("negativePrompt", "")
    if not isinstance(negative_prompt, str):
        raise ValueError("negativePrompt must be a string")
    normalized["negativePrompt"] = negative_prompt
    configuration = normalized.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("configuration must be an object")
    normalized["configuration"] = _json_value(configuration, "configuration")
    for key in ("inputs", "loras"):
        collection = normalized.get(key, [])
        if not isinstance(collection, list):
            raise ValueError(f"{key} must be an array")
        normalized[key] = _json_value(collection, key)
    _json_value(normalized, "request")
    return normalized


def request_fingerprint(value: dict[str, Any]) -> str:
    normalized = validate_request(value)
    identity = {key: item for key, item in normalized.items() if key not in _TRANSIENT_REQUEST_KEYS}
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_event(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("event must be an object")
    normalized = dict(value)
    _nonempty_string(normalized, "requestID")
    event_type = normalized.get("type")
    if not isinstance(event_type, str) or event_type not in EVENT_TYPES:
        raise ValueError("type must be progress, preview, result, or error")
    _json_value(normalized, "event")
    return normalized
