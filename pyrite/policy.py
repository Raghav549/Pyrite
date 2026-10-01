from __future__ import annotations

from dataclasses import dataclass

from .privacy import PrivacyPolicy


@dataclass(frozen=True)
class DevicePolicy:
    ram_budget_mb: int
    working_set_mb: int
    privacy: PrivacyPolicy = PrivacyPolicy()

    def validate(self) -> None:
        if self.ram_budget_mb < 1024:
            raise ValueError("Pyrite requires at least a 1 GiB configured RAM budget")
        if self.working_set_mb <= 0:
            raise ValueError("working set must be positive")
        self.privacy.validate()
