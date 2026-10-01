from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class RuntimeConfig:
    ram_budget_mb: int = 4096
    reserve_mb: int = 768
    max_context_tokens: int = 2048
    max_kv_tokens: int = 1536
    storage_dir: Path = Path(".pyrite/models")
    prefetch_depth: int = 2
    resident_blocks: int = 2
    offline: bool = True

    @property
    def working_set_mb(self) -> int:
        return max(256, self.ram_budget_mb - self.reserve_mb)

    @property
    def resident_byte_budget(self) -> int:
        return self.working_set_mb * 1024 * 1024

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        return cls(
            ram_budget_mb=int(os.getenv("PYRITE_RAM_MB", "4096")),
            reserve_mb=int(os.getenv("PYRITE_RESERVE_MB", "768")),
            max_context_tokens=int(os.getenv("PYRITE_CONTEXT", "2048")),
            max_kv_tokens=int(os.getenv("PYRITE_KV_TOKENS", "1536")),
            storage_dir=Path(os.getenv("PYRITE_MODEL_DIR", ".pyrite/models")),
            prefetch_depth=int(os.getenv("PYRITE_PREFETCH", "2")),
            resident_blocks=int(os.getenv("PYRITE_RESIDENT_BLOCKS", "2")),
            offline=os.getenv("PYRITE_OFFLINE", "1") != "0",
        )
