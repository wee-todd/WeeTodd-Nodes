from __future__ import annotations

from dataclasses import asdict, dataclass
from urllib.parse import unquote


@dataclass(frozen=True)
class DrawThingsProfile:
    """A connection route kept separate from an endpoint model selection."""

    id: str
    name: str
    route: str
    host: str
    port: int
    useTLS: bool
    credentialRef: str | None = None
    selfHostedConfirmed: bool = False

    def __post_init__(self) -> None:
        for key in ("id", "name", "host"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"{key} must be a non-empty string")
        if not isinstance(self.route, str) or self.route not in {"grpc", "dtBridge", "dtCloud"}:
            raise ValueError("route must be grpc, dtBridge, or dtCloud")
        decoded_host = unquote(self.host)
        if any(character.isspace() for character in decoded_host) or any(
            marker in decoded_host for marker in ("@", "/", "\\", "?", "#")
        ):
            raise ValueError("host must be a bare hostname or IP address without credentials")
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 1 <= self.port <= 65535
        ):
            raise ValueError("port must be an integer from 1 to 65535")
        if not isinstance(self.useTLS, bool):
            raise ValueError("useTLS must be a boolean")
        if not isinstance(self.selfHostedConfirmed, bool):
            raise ValueError("selfHostedConfirmed must be a boolean")
        if self.selfHostedConfirmed and self.route != "grpc":
            raise ValueError("only grpc connections can be confirmed self-hosted")
        if self.credentialRef is not None and (
            not isinstance(self.credentialRef, str) or not self.credentialRef.strip()
        ):
            raise ValueError("credentialRef must be a non-empty string when present")

    def to_dict(self) -> dict[str, str | int | bool | None]:
        return asdict(self)
