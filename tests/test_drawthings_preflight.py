import copy
import math

import pytest

from wee_todd_remote.contracts import request_fingerprint
from wee_todd_remote.preflight import discover, prepare


@pytest.fixture
def remote_request():
    return {
        "schema": "weetodd-drawthings-request-v1",
        "requestID": "request-1",
        "operation": "video",
        "profileID": "cloud",
        "modelID": "ltx-fixture",
        "prompt": "A paper bird takes flight",
        "negativePrompt": "",
        "configuration": {
            "width": 768,
            "height": 512,
            "numFrames": 121,
            "steps": "automatic",
        },
        "inputs": [{"role": "firstFrame", "assetID": "still-1"}],
        "loras": [],
        "billingPolicy": "freeOnly",
    }


@pytest.fixture
def capabilities(remote_request):
    resolved = copy.deepcopy(remote_request)
    resolved["configuration"]["steps"] = 20
    fingerprint = request_fingerprint(resolved)
    return {
        "route": "dtCloud",
        "executionMode": "cloud",
        "confidence": "verified",
        "models": {
            "ltx-fixture": {
                "operations": {
                    "video": {
                        "width": {"min": 512, "max": 1024, "multipleOf": 64},
                        "height": {"min": 512, "max": 1024, "multipleOf": 64},
                        "numFrames": {"min": 1, "max": 121, "multipleOf": 8, "offset": 1},
                        "inputRoleCombinations": [["firstFrame"], ["firstFrame", "lastFrame"]],
                        "maxLoRAs": 1,
                        "automaticSettings": {"steps": 20},
                    }
                }
            }
        },
        "estimate": {
            "cu": 9999,
            "fingerprint": fingerprint,
            "estimatorRevision": "fixture-r1",
        },
    }


@pytest.fixture
def account():
    return {
        "authenticated": True,
        "limitMode": "cloud",
        "limitCU": 10000,
        "policyExpiresAt": 200,
        "billingRoute": "free",
        "routeVerified": True,
    }


@pytest.mark.parametrize("cu,expected", [(9999, "allowed"), (10000, "blocked"), (10001, "blocked")])
def test_verified_policy_boundary(remote_request, capabilities, account, cu, expected):
    capabilities["estimate"]["cu"] = cu
    result = prepare(remote_request, capabilities, account, now=100)
    assert result["eligibility"] == expected
    assert result["estimateCU"] == cu
    assert result["limitCU"] == 10000


def test_prepare_resolves_automatic_settings_before_fingerprinting(
    remote_request, capabilities, account
):
    resolved = copy.deepcopy(remote_request)
    resolved["configuration"]["steps"] = 20
    capabilities["estimate"]["fingerprint"] = request_fingerprint(resolved)
    result = prepare(remote_request, capabilities, account, now=100)
    assert result["normalizedRequest"]["configuration"]["steps"] == 20
    assert result["fingerprint"] == request_fingerprint(resolved)
    assert result["estimateSource"] == "fixture-r1"


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda request: request.update(modelID="unlisted"), "unsupportedModel"),
        (lambda request: request["configuration"].update(width=770), "unsupportedDimensions"),
        (lambda request: request["configuration"].update(numFrames=120), "unsupportedFrameCount"),
        (
            lambda request: request["inputs"].append({"role": "mask", "assetID": "mask-1"}),
            "unsupportedInputs",
        ),
        (lambda request: request["loras"].extend([{"id": "a"}, {"id": "b"}]), "unsupportedLoRAs"),
    ],
)
def test_capability_validation_blocks_before_estimate(
    remote_request, capabilities, account, mutate, code
):
    mutate(remote_request)
    result = prepare(remote_request, capabilities, account, now=100)
    assert result["eligibility"] == "blocked"
    assert code in {issue["code"] for issue in result["issues"]}
    assert result["estimateCU"] is None


def test_missing_operation_capability_is_blocked_without_model_name_guess(
    remote_request, capabilities, account
):
    capabilities["models"]["ltx-fixture"]["operations"] = {"image": {}}
    result = prepare(remote_request, capabilities, account, now=100)
    assert result["eligibility"] == "blocked"
    assert result["issues"][0]["code"] == "unsupportedOperation"


def test_stale_estimate_for_changed_request_is_unknown(remote_request, capabilities, account):
    remote_request["prompt"] = "A changed prompt"
    result = prepare(remote_request, capabilities, account, now=100)
    assert result["eligibility"] == "unknown"
    assert result["estimateCU"] is None
    assert "estimateFingerprintMismatch" in {issue["code"] for issue in result["issues"]}


@pytest.mark.parametrize(
    "change,code",
    [
        ({"routeVerified": False}, "billingRouteUnverified"),
        ({"billingRoute": "unknown"}, "billingRouteUnknown"),
        ({"limitCU": None}, "limitUnknown"),
        ({"policyExpiresAt": 100}, "policyExpired"),
    ],
)
def test_incomplete_or_expired_cloud_policy_is_unknown(
    remote_request, capabilities, account, change, code
):
    account.update(change)
    result = prepare(remote_request, capabilities, account, now=100)
    assert result["eligibility"] == "unknown"
    assert code in {issue["code"] for issue in result["issues"]}


def test_confirmed_invalid_auth_is_blocked(remote_request, capabilities, account):
    account["authenticated"] = False
    result = prepare(remote_request, capabilities, account, now=100)
    assert result["eligibility"] == "blocked"
    assert "authenticationRejected" in {issue["code"] for issue in result["issues"]}


@pytest.mark.parametrize("billing_route", ["paid", "boost"])
def test_free_only_rejects_verified_nonfree_billing(
    remote_request, capabilities, account, billing_route
):
    account["billingRoute"] = billing_route
    result = prepare(remote_request, capabilities, account, now=100)
    assert result["eligibility"] == "blocked"
    assert "freeOnlyBillingRejected" in {issue["code"] for issue in result["issues"]}


def test_verified_selfhost_does_not_require_cloud_account_or_estimate(remote_request, capabilities):
    capabilities.update(route="grpc", executionMode="selfHosted", estimate=None)
    local_account = {
        "authenticated": False,
        "limitMode": "notApplicable",
        "limitCU": None,
        "policyExpiresAt": None,
        "billingRoute": "unknown",
        "routeVerified": True,
    }
    result = prepare(remote_request, capabilities, local_account, now=100)
    assert result["eligibility"] == "allowed"
    assert result["limitMode"] == "notApplicable"
    assert result["estimateCU"] is None


def test_localhost_bridge_cannot_claim_cloud_limit_not_applicable(remote_request, capabilities):
    capabilities.update(route="dtBridge", executionMode="bridge", estimate=None)
    bridge_account = {
        "authenticated": False,
        "limitMode": "notApplicable",
        "limitCU": None,
        "policyExpiresAt": None,
        "billingRoute": "unknown",
        "routeVerified": True,
    }
    result = prepare(remote_request, capabilities, bridge_account, now=100)
    assert result["eligibility"] == "blocked"
    assert "cloudPolicyRequired" in {issue["code"] for issue in result["issues"]}


@pytest.mark.parametrize(
    "target,key,bad",
    [
        ("estimate", "cu", True),
        ("estimate", "cu", math.inf),
        ("account", "limitCU", "10000"),
        ("account", "policyExpiresAt", math.nan),
        ("account", "authenticated", 1),
        ("account", "routeVerified", "yes"),
    ],
)
def test_wrong_typed_policy_numbers_and_bools_are_unknown(
    remote_request, capabilities, account, target, key, bad
):
    (capabilities["estimate"] if target == "estimate" else account)[key] = bad
    result = prepare(remote_request, capabilities, account, now=100)
    assert result["eligibility"] == "unknown"
    assert result["issues"]


def test_malformed_request_returns_structured_blocked_result(capabilities, account):
    result = prepare({"schema": "wrong"}, capabilities, account, now=100)
    assert result["eligibility"] == "blocked"
    assert result["normalizedRequest"] is None
    assert result["issues"][0]["code"] == "invalidRequest"


def test_discover_normalizes_explicit_profile_snapshot_without_guessing():
    snapshot = discover(
        {
            "route": "grpc",
            "executionMode": "selfHosted",
            "confidence": "verified",
            "models": {"exact-id": {"operations": {"image": {}}}},
        }
    )
    assert snapshot == {
        "route": "grpc",
        "executionMode": "selfHosted",
        "confidence": "verified",
        "models": {"exact-id": {"operations": {"image": {}}}},
        "estimate": None,
    }


def test_discover_rejects_unverified_or_malformed_advertisements():
    with pytest.raises(ValueError, match="confidence"):
        discover({"route": "grpc", "executionMode": "selfHosted", "models": {}})
    with pytest.raises(ValueError, match="models"):
        discover(
            {"route": "grpc", "executionMode": "selfHosted", "confidence": "verified", "models": []}
        )


@pytest.mark.parametrize("field", ["route", "executionMode"])
def test_discover_rejects_unhashable_connection_metadata(field):
    snapshot = {
        "route": "grpc",
        "executionMode": "selfHosted",
        "confidence": "verified",
        "models": {},
    }
    snapshot[field] = []
    with pytest.raises(ValueError, match=field):
        discover(snapshot)


@pytest.mark.parametrize(
    ("route", "execution_mode"),
    [("garbage", "cloud"), ("grpc", "cloud"), ("dtCloud", "selfHosted")],
)
def test_prepare_blocks_invalid_or_contradictory_route_mode(
    remote_request, capabilities, account, route, execution_mode
):
    capabilities.update(route=route, executionMode=execution_mode)
    result = prepare(remote_request, capabilities, account, now=100)
    assert result["eligibility"] == "blocked"
    assert "invalidRouteMode" in {issue["code"] for issue in result["issues"]}


def test_discover_rejects_contradictory_route_mode():
    with pytest.raises(ValueError, match="route.*executionMode"):
        discover(
            {
                "route": "dtCloud",
                "executionMode": "selfHosted",
                "confidence": "verified",
                "models": {},
            }
        )


@pytest.mark.parametrize(
    "automatic_settings",
    [None, {"steps": math.nan}, {"nested": {"bad": object()}}, {"token": "secret"}],
)
def test_prepare_blocks_malformed_automatic_settings(
    remote_request, capabilities, account, automatic_settings
):
    operation = capabilities["models"]["ltx-fixture"]["operations"]["video"]
    operation["automaticSettings"] = automatic_settings
    result = prepare(remote_request, capabilities, account, now=100)
    assert result["eligibility"] == "blocked"
    assert "invalidAutomaticSettings" in {issue["code"] for issue in result["issues"]}


def test_prepare_blocks_unresolved_automatic_configuration(remote_request, capabilities, account):
    operation = capabilities["models"]["ltx-fixture"]["operations"]["video"]
    operation["automaticSettings"] = {}
    result = prepare(remote_request, capabilities, account, now=100)
    assert result["eligibility"] == "blocked"
    assert "unresolvedAutomaticSetting" in {issue["code"] for issue in result["issues"]}


@pytest.mark.parametrize("now", ["100", math.nan, math.inf, True])
def test_prepare_returns_unknown_for_invalid_clock(remote_request, capabilities, account, now):
    result = prepare(remote_request, capabilities, account, now=now)
    assert result["eligibility"] == "unknown"
    assert "clockUnknown" in {issue["code"] for issue in result["issues"]}


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("limitMode", [], "cloudPolicyRequired"),
        ("limitMode", {}, "cloudPolicyRequired"),
        ("billingRoute", [], "billingRouteUnknown"),
        ("billingRoute", {}, "billingRouteUnknown"),
    ],
)
def test_prepare_handles_unhashable_account_enums_as_unknown(
    remote_request, capabilities, account, field, value, code
):
    account[field] = value
    result = prepare(remote_request, capabilities, account, now=100)
    assert result["eligibility"] == "unknown"
    assert code in {issue["code"] for issue in result["issues"]}
