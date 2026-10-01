from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Memory and execution budget for a local Pyrite runtime.

    ``ram_budget_mb`` is the *resident* budget Pyrite is allowed to use, not a
    claim that a model of that size fits in RAM.  ``reserve_mb`` is held back for
    the interpreter, tokenizer, KV cache and the operating system.
    """

    ram_budget_mb: int = 4096
    reserve_mb: int = 768
    max_context_tokens: int = 2048
    max_kv_tokens: int = 1536
    storage_dir: Path = Path(".pyrite/models")
    prefetch_depth: int = 2
    resident_blocks: int = 2
    offline: bool = True

    def __post_init__(self) -> None:
        # Accept strings for storage_dir from JSON/env profiles.
        if not isinstance(self.storage_dir, Path):
            object.__setattr__(self, "storage_dir", Path(self.storage_dir))

    @property
    def working_set_mb(self) -> int:
        return self.ram_budget_mb - self.reserve_mb

    @property
    def resident_byte_budget(self) -> int:
        return self.working_set_mb * 1024 * 1024

    def validate(self) -> None:
        if self.ram_budget_mb < 1024:
            raise ValueError("ram_budget_mb must be at least 1024")
        if self.reserve_mb < 0:
            raise ValueError("reserve_mb must be non-negative")
        if self.working_set_mb < 256:
            raise ValueError(
                "reserve_mb leaves less than 256 MiB of the RAM budget for the working set"
            )
        if self.resident_blocks < 1:
            raise ValueError("resident_blocks must be at least 1")
        if self.prefetch_depth < 0:
            raise ValueError("prefetch_depth must be non-negative")
        if self.max_context_tokens < 1:
            raise ValueError("max_context_tokens must be positive")
        if self.max_kv_tokens < 1:
            raise ValueError("max_kv_tokens must be positive")
        if self.max_kv_tokens > self.max_context_tokens:
            raise ValueError("max_kv_tokens cannot exceed max_context_tokens")

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["storage_dir"] = str(self.storage_dir)
        return data

    def with_overrides(self, **overrides: object) -> RuntimeConfig:
        known = {field.name for field in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"unknown runtime config keys: {', '.join(sorted(unknown))}")
        cleaned = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **cleaned)

    #: Profile keys that are documentation rather than configuration.
    PROFILE_METADATA_KEYS = frozenset({"name", "description", "comment"})

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RuntimeConfig:
        if not isinstance(data, dict):
            raise TypeError("runtime profile must be a JSON object")
        base = cls()
        keys = {field.name for field in fields(cls)}
        unknown = set(data) - keys - cls.PROFILE_METADATA_KEYS
        if unknown:
            raise ValueError(f"unknown runtime config keys: {', '.join(sorted(unknown))}")
        return base.with_overrides(**{key: value for key, value in data.items() if key in keys})

    @classmethod
    def from_profile(cls, path: str | Path) -> RuntimeConfig:
        """Load a JSON profile such as ``examples/four_gb.json``."""
        profile_path = Path(path).expanduser()
        try:
            data = json.loads(profile_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"runtime profile not found: {profile_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"runtime profile is not valid JSON: {profile_path}") from exc
        return cls.from_dict(data)

    def save_profile(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> RuntimeConfig:
        source = os.environ if env is None else env

        profile = source.get("PYRITE_PROFILE")
        base = cls.from_profile(profile) if profile else cls()

        def number(name: str, default: int) -> int:
            raw = source.get(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer, got {raw!r}") from exc

        storage = source.get("PYRITE_MODEL_DIR")
        return base.with_overrides(
            ram_budget_mb=number("PYRITE_RAM_MB", base.ram_budget_mb),
            reserve_mb=number("PYRITE_RESERVE_MB", base.reserve_mb),
            max_context_tokens=number("PYRITE_CONTEXT", base.max_context_tokens),
            max_kv_tokens=number("PYRITE_KV_TOKENS", base.max_kv_tokens),
            storage_dir=Path(storage) if storage else base.storage_dir,
            prefetch_depth=number("PYRITE_PREFETCH", base.prefetch_depth),
            resident_blocks=number("PYRITE_RESIDENT_BLOCKS", base.resident_blocks),
            offline=source.get("PYRITE_OFFLINE", "1" if base.offline else "0") != "0",
        )
