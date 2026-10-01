from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DevicePolicy:
    ram_budget_mb: int
    working_set_mb: int
    offline_only: bool = True
    allow_network: bool = False

    def validate(self) -> None:
        if self.ram_budget_mb < 1024:
            raise ValueError("Pyrite requires at least a 1 GiB configured RAM budget")
        if self.working_set_mb <= 0:
            raise ValueError("working set must be positive")
        if self.offline_only and self.allow_network:
            raise ValueError("offline-only mode cannot allow network access")
