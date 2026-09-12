"""Pure Draw Things capability and compute-policy preflight.

``discover`` accepts an adapter-produced verified snapshot with this shape::

    {"route": "grpc|dtBridge|dtCloud",
     "executionMode": "selfHosted|bridge|cloud",
     "confidence": "verified",
     "models": {modelID: {"operations": {operation: operationSpec}}},
     "estimate": {"cu": number, "fingerprint": str,
                  "estimatorRevision": str} | None}

An operation spec is the verified intersection of endpoint model discovery
with explicit adapter rules. It contains ``width``, ``height``, and (for
video) ``numFrames`` constraints as ``min``, ``max``, ``multipleOf``, plus an
optional ``offset``. It also provides ``inputRoleCombinations`` (exact role
sets), ``maxLoRAs``, and ``automaticSettings``. Missing capabilities are not
inferred from a model name. Automatic configuration values are replaced only
from the verified settings, producing the one canonical request used for both
fingerprinting and submission.

``prepare`` separately consumes account state with ``authenticated``,
``limitMode`` (cloud/notApplicable/unknown), ``limitCU``, ``policyExpiresAt``,
``billingRoute`` (free/paid/boost/unknown), and ``routeVerified``. CU is an
estimate for this request; ``limitCU`` is the verified per-job boundary. No
monthly quota, balance, entitlement, or production default limit is inferred.
For direct cloud only, a verified ``limitEnforcement: server`` account can omit
the numerical CU limit when PAYG is explicitly disabled and a positive monthly
free allowance is present. Generation still requires fresh provider authorization.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from .contracts import request_fingerprint, validate_request

_ROUTES = frozenset({"grpc", "dtBridge", "dtCloud"})
_EXECUTION_MODES = frozenset({"selfHosted", "bridge", "cloud"})
_ROUTE_MODES = {
    "grpc": "selfHosted",
    "dtBridge": "bridge",
    "dtCloud": "cloud",
}


def _issue(code: str, field: str, message: str) -> dict[str, str]:
    return {"code": code, "field": field, "message": message}


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _nonnegative_number(value: Any) -> bool:
    return _number(value) and value >= 0


def discover(profile: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize an explicit discovery snapshot without inference."""
    if not isinstance(profile, dict):
        raise ValueError("profile must be an object")
    route = profile.get("route")
    if not isinstance(route, str) or route not in _ROUTES:
        raise ValueError("route must be grpc, dtBridge, or dtCloud")
    execution_mode = profile.get("executionMode")
    if not isinstance(execution_mode, str) or execution_mode not in _EXECUTION_MODES:
        raise ValueError("executionMode must be selfHosted, bridge, or cloud")
    if _ROUTE_MODES[route] != execution_mode:
        raise ValueError("route and executionMode must be a consistent pair")
    if profile.get("confidence") != "verified":
        raise ValueError("confidence must be verified")
    models = profile.get("models")
    if not isinstance(models, dict):
        raise ValueError("models must be an object")
    for model_id, model in models.items():
        if not isinstance(model_id, str) or not model_id or not isinstance(model, dict):
            raise ValueError("models must map non-empty model IDs to objects")
        operations = model.get("operations")
        if not isinstance(operations, dict):
            raise ValueError("models operations must be an object")
        if any(
            operation not in {"image", "video"} or not isinstance(spec, dict)
            for operation, spec in operations.items()
        ):
            raise ValueError("models operations contain an invalid operation")
    return {
        "route": route,
        "executionMode": execution_mode,
        "confidence": "verified",
        "models": copy.deepcopy(models),
        "estimate": copy.deepcopy(profile.get("estimate")),
    }


def _result(normalized: dict[str, Any] | None, fingerprint: str | None) -> dict[str, Any]:
    return {
        "normalizedRequest": normalized,
        "fingerprint": fingerprint,
        "estimateCU": None,
        "limitCU": None,
        "estimateSource": None,
        "policyExpiresAt": None,
        "limitMode": "unknown",
        "eligibility": "unknown",
        "issues": [],
    }


def _validate_constraint(value: Any, constraint: Any) -> bool:
    if not isinstance(value, int) or isinstance(value, bool) or not isinstance(constraint, dict):
        return False
    minimum, maximum = constraint.get("min"), constraint.get("max")
    multiple, offset = constraint.get("multipleOf", 1), constraint.get("offset", 0)
    if not all(
        isinstance(item, int) and not isinstance(item, bool)
        for item in (minimum, maximum, multiple, offset)
    ):
        return False
    return minimum <= value <= maximum and multiple > 0 and (value - offset) % multiple == 0


def _canonical_request(request: dict[str, Any], operation_spec: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(request)
    automatic = operation_spec.get("automaticSettings")
    if not isinstance(automatic, dict):
        raise ValueError("automaticSettings must be an object")
    probe = copy.deepcopy(request)
    probe["configuration"] = copy.deepcopy(automatic)
    validate_request(probe)
    configuration = normalized["configuration"]
    for key, value in automatic.items():
        if key not in configuration or configuration[key] == "automatic":
            configuration[key] = copy.deepcopy(value)
    if _contains_automatic(configuration):
        raise ValueError("configuration contains an unresolved automatic setting")
    return validate_request(normalized)


def _contains_automatic(value: Any) -> bool:
    if value == "automatic":
        return True
    if isinstance(value, list):
        return any(_contains_automatic(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_automatic(item) for item in value.values())
    return False


def _capability_issues(
    request: dict[str, Any], capabilities: Any
) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    issues: list[dict[str, str]] = []
    if not isinstance(capabilities, dict) or capabilities.get("confidence") != "verified":
        return [
            _issue("capabilitiesUnverified", "capabilities", "verified capabilities are required")
        ], None
    route = capabilities.get("route")
    execution_mode = capabilities.get("executionMode")
    if (
        not isinstance(route, str)
        or not isinstance(execution_mode, str)
        or _ROUTE_MODES.get(route) != execution_mode
    ):
        return [
            _issue(
                "invalidRouteMode",
                "capabilities.route",
                "route and executionMode must be a defined consistent pair",
            )
        ], None
    models = capabilities.get("models")
    model = models.get(request["modelID"]) if isinstance(models, dict) else None
    if not isinstance(model, dict):
        return [_issue("unsupportedModel", "modelID", "the exact model is not advertised")], None
    operations = model.get("operations")
    spec = operations.get(request["operation"]) if isinstance(operations, dict) else None
    if not isinstance(spec, dict):
        return [
            _issue(
                "unsupportedOperation", "operation", "the model does not advertise this operation"
            )
        ], None
    configuration = request["configuration"]
    for key in ("width", "height"):
        if not _validate_constraint(configuration.get(key), spec.get(key)):
            issues.append(
                _issue(
                    "unsupportedDimensions",
                    f"configuration.{key}",
                    f"{key} is outside the advertised constraint",
                )
            )
    if request["operation"] == "video" and not _validate_constraint(
        configuration.get("numFrames"), spec.get("numFrames")
    ):
        issues.append(
            _issue(
                "unsupportedFrameCount",
                "configuration.numFrames",
                "numFrames is outside the verified constraint",
            )
        )
    combinations = spec.get("inputRoleCombinations")
    roles = []
    valid_roles = True
    for item in request["inputs"]:
        if not isinstance(item, dict) or not isinstance(item.get("role"), str):
            valid_roles = False
            break
        roles.append(item["role"])
    advertised = isinstance(combinations, list) and any(
        isinstance(combo, list)
        and all(isinstance(role, str) for role in combo)
        and sorted(combo) == sorted(roles)
        for combo in combinations
    )
    if not valid_roles or not advertised:
        issues.append(
            _issue(
                "unsupportedInputs", "inputs", "the exact input role combination is not advertised"
            )
        )
    max_loras = spec.get("maxLoRAs")
    if (
        not isinstance(max_loras, int)
        or isinstance(max_loras, bool)
        or max_loras < 0
        or len(request["loras"]) > max_loras
    ):
        issues.append(
            _issue(
                "unsupportedLoRAs", "loras", "the LoRA count exceeds or lacks an advertised limit"
            )
        )
    return issues, spec


def prepare(
    request: dict[str, Any], capabilities: dict[str, Any], account: dict[str, Any], now: float
) -> dict[str, Any]:
    """Return a pure, structured generation eligibility decision."""
    try:
        normalized = validate_request(request)
    except (TypeError, ValueError) as error:
        result = _result(None, None)
        result.update(
            eligibility="blocked", issues=[_issue("invalidRequest", "request", str(error))]
        )
        return result
    issues, operation_spec = _capability_issues(normalized, capabilities)
    if issues:
        result = _result(normalized, request_fingerprint(normalized))
        result.update(eligibility="blocked", issues=issues)
        return result
    assert operation_spec is not None
    try:
        normalized = _canonical_request(normalized, operation_spec)
    except (TypeError, ValueError) as error:
        result = _result(normalized, request_fingerprint(normalized))
        code = (
            "unresolvedAutomaticSetting"
            if "unresolved automatic" in str(error)
            else "invalidAutomaticSettings"
        )
        result.update(
            eligibility="blocked",
            issues=[_issue(code, "capabilities.automaticSettings", str(error))],
        )
        return result
    fingerprint = request_fingerprint(normalized)
    result = _result(normalized, fingerprint)
    if not _number(now):
        result["issues"] = [_issue("clockUnknown", "now", "current time must be finite")]
        return result
    if not isinstance(account, dict):
        result["issues"] = [_issue("accountUnknown", "account", "account state is unavailable")]
        return result
    limit_mode = account.get("limitMode")
    result["limitMode"] = (
        limit_mode
        if isinstance(limit_mode, str) and limit_mode in {"cloud", "notApplicable", "unknown"}
        else "unknown"
    )
    result["policyExpiresAt"] = (
        account.get("policyExpiresAt") if _number(account.get("policyExpiresAt")) else None
    )
    route = capabilities.get("route")
    execution_mode = capabilities.get("executionMode")
    if (
        route == "grpc"
        and execution_mode == "selfHosted"
        and account.get("routeVerified") is True
        and result["limitMode"] == "notApplicable"
    ):
        result["eligibility"] = "allowed"
        estimate = capabilities.get("estimate")
        if (
            isinstance(estimate, dict)
            and _nonnegative_number(estimate.get("cu"))
            and estimate.get("fingerprint") == fingerprint
            and isinstance(estimate.get("estimatorRevision"), str)
            and estimate["estimatorRevision"]
        ):
            result["estimateCU"] = estimate["cu"]
            result["estimateSource"] = estimate["estimatorRevision"]
        return result
    if result["limitMode"] != "cloud":
        result["issues"].append(
            _issue(
                "cloudPolicyRequired",
                "account.limitMode",
                "cloud or bridge routes require cloud policy",
            )
        )
    authenticated = account.get("authenticated")
    if authenticated is False:
        result.update(eligibility="blocked")
        result["issues"].append(
            _issue("authenticationRejected", "account.authenticated", "authentication was rejected")
        )
    elif authenticated is not True:
        result["issues"].append(
            _issue(
                "authenticationUnknown", "account.authenticated", "authentication is not confirmed"
            )
        )
    route_verified = account.get("routeVerified")
    if route_verified is not True:
        result["issues"].append(
            _issue(
                "billingRouteUnverified",
                "account.routeVerified",
                "the effective route is not verified",
            )
        )
    billing_route = account.get("billingRoute")
    if (
        route_verified is True
        and isinstance(billing_route, str)
        and billing_route in {"paid", "boost"}
    ):
        result.update(eligibility="blocked")
        result["issues"].append(
            _issue(
                "freeOnlyBillingRejected",
                "account.billingRoute",
                "freeOnly forbids the verified billing route",
            )
        )
    elif billing_route != "free":
        result["issues"].append(
            _issue(
                "billingRouteUnknown",
                "account.billingRoute",
                "a free billing route is not confirmed",
            )
        )
    expires = account.get("policyExpiresAt")
    if not _number(expires):
        result["issues"].append(
            _issue("policyExpiryUnknown", "account.policyExpiresAt", "policy expiry is unavailable")
        )
    elif expires <= now:
        result["issues"].append(
            _issue("policyExpired", "account.policyExpiresAt", "account policy has expired")
        )
    limit = account.get("limitCU")
    if isinstance(account.get("reason"), str):
        result["accountMessage"] = account["reason"]
    quota = account.get("monthlyQuota")
    remaining = quota.get("remainingRequests") if isinstance(quota, dict) else None
    server_limit = (
        limit is None
        and capabilities.get("route") == "dtCloud"
        and capabilities.get("executionMode") == "cloud"
        and account.get("limitEnforcement") == "server"
        and account.get("paygEnabled") is False
        and _number(remaining) and remaining > 0 and remaining == int(remaining)
    )
    if _nonnegative_number(limit):
        result["limitCU"] = limit
    elif server_limit:
        result["limitEnforcement"] = "server"
    else:
        result["issues"].append(
            _issue(
                "limitUnknown", "account.limitCU", "the effective per-job CU limit is unavailable"
            )
        )
    estimate = capabilities.get("estimate")
    if not isinstance(estimate, dict) or not _nonnegative_number(estimate.get("cu")):
        result["issues"].append(
            _issue(
                "estimateUnknown",
                "capabilities.estimate.cu",
                "a finite non-negative CU estimate is unavailable",
            )
        )
    elif estimate.get("fingerprint") != fingerprint:
        result["issues"].append(
            _issue(
                "estimateFingerprintMismatch",
                "capabilities.estimate.fingerprint",
                "estimate does not match the canonical request",
            )
        )
    elif (
        not isinstance(estimate.get("estimatorRevision"), str) or not estimate["estimatorRevision"]
    ):
        result["issues"].append(
            _issue(
                "estimateSourceUnknown",
                "capabilities.estimate.estimatorRevision",
                "estimator revision is unavailable",
            )
        )
    else:
        result["estimateCU"] = estimate["cu"]
        result["estimateSource"] = estimate["estimatorRevision"]
    if result["eligibility"] == "blocked":
        return result
    if not result["issues"] and result["estimateCU"] is not None and server_limit:
        result["eligibility"] = "allowed"
    elif (not result["issues"] and result["estimateCU"] is not None
          and result["limitCU"] is not None):
        result["eligibility"] = "allowed" if result["estimateCU"] < result["limitCU"] else "blocked"
        if result["eligibility"] == "blocked":
            result["issues"].append(
                _issue(
                    "cuLimitReached", "estimateCU", "estimate meets or exceeds the per-job CU limit"
                )
            )
    return result
