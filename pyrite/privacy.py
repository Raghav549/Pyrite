from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrivacyPolicy:
    offline_only: bool = True
    telemetry: bool = False
    remote_inference: bool = False

    def validate(self) -> None:
        if self.offline_only and (self.telemetry or self.remote_inference):
            raise ValueError("offline-only Pyrite cannot enable telemetry or remote inference")

    def assert_network_allowed(self) -> None:
        if self.offline_only:
            raise PermissionError("Pyrite is configured for offline-only operation")
