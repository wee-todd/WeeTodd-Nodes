"""Shared Draw Things helper adapter with fresh, free-only preflight per generation."""

from __future__ import annotations

import copy
import json
import uuid
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .client import invoke_helper
from .conditioning import validate_canonical_inputs, validate_discovered_loras
from .contracts import request_fingerprint, validate_request
from .media import validate_media
from .preflight import prepare as prepare_policy


class DrawThingsAdapter:
    """Coordinate isolated helper calls without retaining credentials or policy state.

    ``credential_provider(profile)`` returns an ephemeral dictionary.
    ``account_provider(profile, discovery)`` may return verified effective account policy.
    Neither provider's result is stored on the adapter.
    """

    def __init__(
        self,
        *,
        helper: Path,
        profiles: Mapping[str, Any] | Callable[[str], Any],
        credential_provider: Callable[[Any], dict[str, Any]],
        account_provider: Callable[[Any, dict[str, Any]], dict[str, Any]] | None = None,
        invoke: Callable[..., Iterator[dict[str, Any]]] = invoke_helper,
        now: Callable[[], float] | float,
    ) -> None:
        self._helper = Path(helper)
        self._profiles = profiles
        self._credential_provider = credential_provider
        self._account_provider = account_provider
        self._invoke = invoke
        self._now = now

    def _profile(self, profile_id: str) -> Any:
        if not isinstance(profile_id, str) or not profile_id:
            raise ValueError("profileID must identify a saved Draw Things profile")
        try:
            profile = (
                self._profiles(profile_id)
                if callable(self._profiles)
                else self._profiles[profile_id]
            )
        except (KeyError, LookupError) as error:
            raise ValueError("unknown Draw Things profileID") from error
        if profile is None or getattr(profile, "id", None) != profile_id:
            raise ValueError("profile resolver returned a mismatched Draw Things profile")
        return profile

    def _credentials(self, profile: Any) -> dict[str, Any]:
        try:
            value = self._credential_provider(profile)
        except Exception as error:
            raise RuntimeError("Draw Things credentials are unavailable") from error
        if not isinstance(value, dict):
            raise RuntimeError("Draw Things credentials are unavailable")
        return copy.deepcopy(value)

    def _profile_payload(self, profile: Any) -> dict[str, Any]:
        value = profile.to_dict()
        if not isinstance(value, dict):
            raise ValueError("Draw Things profile must serialize to an object")
        return copy.deepcopy(value)

    def _call(
        self,
        command: str,
        payload: dict[str, Any],
        cancelled: Callable[[], bool],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        progress: list[dict[str, Any]] = []
        result = None
        for event in self._invoke(command, payload, helper=self._helper, cancelled=cancelled):
            if event.get("type") == "result":
                result = event
            else:
                progress.append(event)
        if not isinstance(result, dict) or not isinstance(result.get("value"), dict):
            raise RuntimeError("Draw Things helper returned no structured result")
        return progress, result

    def _discover(self, profile: Any, cancelled: Callable[[], bool]) -> dict[str, Any]:
        request_id = f"discovery-{uuid.uuid4().hex}"
        payload = {
            "requestID": request_id,
            "profile": self._profile_payload(profile),
            "credentials": self._credentials(profile),
        }
        _, event = self._call("capabilities", payload, cancelled)
        return copy.deepcopy(event["value"])

    def discover(self, profile_id: str) -> dict[str, Any]:
        """Refresh and return helper discovery for one exact saved profile."""
        return self._discover(self._profile(profile_id), lambda: False)

    @staticmethod
    def _capabilities(profile: Any, discovery: dict[str, Any]) -> dict[str, Any]:
        entries = discovery.get("capabilities")
        inventory = discovery.get("files")
        endpoint_ids = (
            set(inventory)
            if isinstance(inventory, list) and all(isinstance(item, str) for item in inventory)
            else set()
        )
        models: dict[str, Any] = {}
        confidence = "verified"
        if not isinstance(entries, dict):
            confidence = "unknown"
        else:
            for model_id, entry in entries.items():
                if not isinstance(model_id, str) or not isinstance(entry, dict):
                    confidence = "unknown"
                    continue
                if model_id not in endpoint_ids:
                    continue
                if (
                    entry.get("confidence") != "verified"
                    or entry.get("source") != "adapter-rules-intersect-endpoint-files"
                    or not isinstance(entry.get("revision"), str)
                    or not entry["revision"]
                ):
                    continue
                models[model_id] = {"operations": copy.deepcopy(entry.get("operations"))}
        route = getattr(profile, "route", None)
        execution_mode = {"grpc": "selfHosted", "dtBridge": "bridge", "dtCloud": "cloud"}.get(route)
        return {
            "route": route,
            "executionMode": execution_mode,
            "confidence": confidence,
            "models": models,
            "estimate": None,
        }

    def _account(self, profile: Any, discovery: dict[str, Any]) -> dict[str, Any]:
        authenticated = discovery.get("authenticated")
        if authenticated not in (True, False):
            transport = discovery.get("transport")
            authenticated = transport.get("authenticated") if isinstance(transport, dict) else None
        if (
            getattr(profile, "route", None) == "grpc"
            and getattr(profile, "selfHostedConfirmed", False) is True
            and authenticated is True
        ):
            return {
                "authenticated": True,
                "limitMode": "notApplicable",
                "limitCU": None,
                "policyExpiresAt": None,
                "billingRoute": "unknown",
                "routeVerified": True,
            }
        if self._account_provider is not None:
            try:
                value = self._account_provider(profile, copy.deepcopy(discovery))
            except Exception as error:
                raise RuntimeError("Draw Things account policy is unavailable") from error
            if isinstance(value, dict):
                return copy.deepcopy(value)
        discovered_account = discovery.get("account")
        if isinstance(discovered_account, dict):
            return copy.deepcopy(discovered_account)
        thresholds = discovery.get("thresholds")
        thresholds = thresholds if isinstance(thresholds, dict) else discovery
        return {
            "authenticated": authenticated,
            "limitMode": thresholds.get("limitMode", "unknown"),
            "limitCU": thresholds.get("limitCU"),
            "policyExpiresAt": thresholds.get("policyExpiresAt"),
            "billingRoute": thresholds.get("billingRoute", "unknown"),
            "routeVerified": thresholds.get("routeVerified", False),
        }

    def _now_value(self) -> Any:
        return self._now() if callable(self._now) else self._now

    def _request_payload(
        self,
        canonical: dict[str, Any],
        profile: Any,
        *,
        output_directory: Path | None = None,
    ) -> dict[str, Any]:
        payload = copy.deepcopy(canonical)
        payload["profile"] = self._profile_payload(profile)
        payload["credentials"] = self._credentials(profile)
        if output_directory is not None:
            payload["outputDirectory"] = str(output_directory)
        return payload

    @staticmethod
    def _adopt_estimate_configuration(
        canonical: dict[str, Any], configuration: Any
    ) -> dict[str, Any] | None:
        if not isinstance(configuration, dict):
            return None
        allowed = {
            "width",
            "height",
            "steps",
            "seed",
            "guidanceScale",
            "strength",
            "shift",
            "sampler",
        }
        if canonical["operation"] == "video":
            allowed.update({"numFrames", "fps"})
        if any(not isinstance(key, str) or key not in allowed for key in configuration):
            return None
        explicit = canonical["configuration"]
        if any(
            key not in configuration or configuration[key] != value
            for key, value in explicit.items()
        ):
            return None
        candidate = copy.deepcopy(canonical)
        candidate["configuration"] = copy.deepcopy(configuration)
        try:
            return validate_request(candidate)
        except (TypeError, ValueError):
            return None

    def _prepare(
        self, request: dict[str, Any], cancelled: Callable[[], bool]
    ) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, Any]]:
        normalized = validate_request(request)
        validate_canonical_inputs(normalized)
        normalized.pop("account", None)
        normalized.pop("estimate", None)
        profile = self._profile(normalized["profileID"])
        discovery = self._discover(profile, cancelled)
        validate_discovered_loras(normalized, discovery)
        capabilities = self._capabilities(profile, discovery)
        unresolved_account = {
            "authenticated": None,
            "limitMode": "unknown",
            "limitCU": None,
            "policyExpiresAt": None,
            "billingRoute": "unknown",
            "routeVerified": False,
        }
        initial = prepare_policy(normalized, capabilities, unresolved_account, self._now_value())
        canonical = initial.get("normalizedRequest")
        if initial["eligibility"] == "blocked" or not isinstance(canonical, dict):
            return initial, discovery, profile, capabilities

        estimate_payload = self._request_payload(canonical, profile)
        _, estimate_event = self._call("estimate", estimate_payload, cancelled)
        estimate = estimate_event["value"]
        adopted = self._adopt_estimate_configuration(canonical, estimate.get("configuration"))
        if adopted is not None:
            canonical = adopted
        fingerprint = (
            request_fingerprint(canonical)
            if adopted is not None
            else "mismatched-estimate-configuration"
        )
        capabilities["estimate"] = {
            "cu": estimate.get("cu"),
            "fingerprint": fingerprint,
            "estimatorRevision": estimate.get("estimatorRevision"),
        }
        account = self._account(profile, discovery)
        final = prepare_policy(canonical, capabilities, account, self._now_value())
        return final, discovery, profile, capabilities

    def prepare(self, request: dict[str, Any]) -> dict[str, Any]:
        """Refresh discovery, estimate, and account state and return preflight."""
        result, _, _, _ = self._prepare(request, lambda: False)
        return result

    def generate(
        self,
        request: dict[str, Any],
        output_directory: Path,
        cancelled: Callable[[], bool],
    ) -> Iterator[dict[str, Any]]:
        """Preflight freshly, stream generation, then validate request-owned media."""
        if cancelled():
            raise InterruptedError("Draw Things request cancelled")
        output_root = Path(output_directory).expanduser().resolve()
        if output_root.exists():
            raise ValueError("outputDirectory must be a new request-owned directory")
        prepared, discovery, profile, _ = self._prepare(request, cancelled)
        if prepared["eligibility"] != "allowed":
            raise RuntimeError(f"Draw Things preflight {prepared['eligibility']}")
        canonical = prepared["normalizedRequest"]
        output_root.parent.mkdir(parents=True, exist_ok=True)
        payload = self._request_payload(canonical, profile, output_directory=output_root)
        event = None
        for helper_event in self._invoke(
            "generate", payload, helper=self._helper, cancelled=cancelled
        ):
            if helper_event.get("type") == "result":
                event = helper_event
            else:
                yield helper_event
        if not isinstance(event, dict) or not isinstance(event.get("value"), dict):
            raise RuntimeError("Draw Things helper returned no structured result")
        manifest_value = event["value"].get("manifestPath")
        if not isinstance(manifest_value, str) or not manifest_value:
            raise ValueError("Draw Things manifestPath must be a non-empty relative path")
        manifest_candidate = Path(manifest_value)
        if not manifest_candidate.is_absolute():
            raise ValueError("Draw Things manifestPath must be an absolute request-owned path")
        try:
            root = output_root.resolve(strict=True)
            manifest_path = manifest_candidate.resolve(strict=True)
        except (FileNotFoundError, RuntimeError) as error:
            raise ValueError("Draw Things manifest does not exist in outputDirectory") from error
        if not manifest_path.is_relative_to(root) or not manifest_path.is_file():
            raise ValueError("Draw Things manifestPath must be confined to outputDirectory")
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError) as error:
            raise ValueError("Draw Things manifest is invalid") from error
        operation = canonical["operation"]
        expected_frames = 1 if operation == "image" else canonical["configuration"].get("numFrames")
        entry = discovery.get("capabilities", {}).get(canonical["modelID"], {})
        operation_spec = entry.get("operations", {}).get(operation, {})
        requires_audio = operation_spec.get("requiresAudio") is True
        media = validate_media(
            manifest,
            root=root,
            expected_request_id=canonical["requestID"],
            expected_operation=operation,
            expected_frames=expected_frames,
            requires_audio=requires_audio,
        )
        result = copy.deepcopy(event)
        result["value"] = {
            "manifestPath": str(manifest_path),
            "media": media,
            "fingerprint": prepared["fingerprint"],
            "normalizedRequest": copy.deepcopy(canonical),
        }
        yield result
