from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import RuntimeConfig
from .kv import KVBudget
from .memory import LRUResidentCache, process_memory_mb
from .policy import DevicePolicy
from .router import HeuristicRouter, Route
from .scheduler import BlockScheduler, LoadPlan
from .storage import LocalBlockStore


@dataclass(frozen=True)
class RuntimeStatus:
    ram_budget_mb: int
    working_set_mb: int
    process_rss_mb: float
    resident_blocks: int
    route: str
    offline: bool


class PyriteRuntime:
    """Orchestration layer for low-memory local inference."""

    def __init__(self, config: RuntimeConfig | None = None):
        self.config = config or RuntimeConfig.from_env()
        self.config.storage_dir.mkdir(parents=True, exist_ok=True)
        self.policy = DevicePolicy(
            self.config.ram_budget_mb,
            self.config.working_set_mb,
            offline_only=self.config.offline,
            allow_network=False,
        )
        self.policy.validate()
        self.store = LocalBlockStore(self.config.storage_dir)
        self.cache = LRUResidentCache[bytes](self.config.resident_blocks)
        self.router = HeuristicRouter()
        self.scheduler = BlockScheduler(self.config.resident_blocks, self.config.prefetch_depth)
        self.kv = KVBudget(self.config.max_kv_tokens)

    def route(self, prompt: str) -> Route:
        return self.router.route(prompt)

    def plan(self, blocks: list[str], current_index: int = 0) -> LoadPlan:
        return self.scheduler.plan(blocks, current_index, set(self.cache.keys()))

    def status(self, route: str = "idle") -> RuntimeStatus:
        return RuntimeStatus(
            self.config.ram_budget_mb,
            self.config.working_set_mb,
            process_memory_mb(),
            self.cache.stats.resident_items,
            route,
            self.config.offline,
        )

    def load_block_bytes(self, block_id: str) -> bytes:
        hit = self.cache.get(block_id)
        if hit is not None:
            return hit
        ref = next((x for x in self.store.iter_blocks() if x.block_id == block_id), None)
        if ref is None:
            raise FileNotFoundError(f"unknown local block: {block_id}")
        payload = ref.path.read_bytes()
        self.cache.put(block_id, payload, len(payload))
        return payload
