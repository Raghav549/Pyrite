from __future__ import annotations

from dataclasses import dataclass

from .config import RuntimeConfig
from .kv import KVBudget
from .kernels.t_sar import ReferenceTernaryKernel
from .memory import LRUResidentCache, process_memory_mb
from .policy import DevicePolicy
from .prefetch import PrefetchPredictor
from .router import HeuristicRouter, Route
from .scheduler import BlockScheduler, LoadPlan
from .storage import BlockRef, LocalBlockStore


@dataclass(frozen=True)
class RuntimeStatus:
    ram_budget_mb: int
    working_set_mb: int
    process_rss_mb: float
    resident_blocks: int
    cache_hits: int
    cache_misses: int
    offline: bool


class PyriteRuntime:
    """Orchestration layer for memory-budgeted local inference."""

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
        self.scheduler = BlockScheduler(
            self.config.resident_blocks,
            self.config.prefetch_depth,
        )
        self.prefetch = PrefetchPredictor()
        self.kv = KVBudget(self.config.max_kv_tokens)
        self.ternary = ReferenceTernaryKernel()

    def route(self, prompt: str) -> Route:
        return self.router.route(prompt)

    def plan(self, blocks: list[str], current_index: int = 0) -> LoadPlan:
        if not blocks:
            return LoadPlan((), (), ())
        current = blocks[current_index] if 0 <= current_index < len(blocks) else blocks[0]
        candidates = blocks[current_index + 1 :]
        predicted = self.prefetch.predict(
            current,
            candidates,
            self.config.prefetch_depth,
        )
        plan = self.scheduler.plan(blocks, current_index, set(self.cache.keys()))
        return LoadPlan(plan.required, predicted or plan.prefetch, plan.evict)

    def load_block_bytes(self, block_id: str) -> bytes:
        hit = self.cache.get(block_id)
        if hit is not None:
            self.prefetch.observe(block_id)
            return hit

        ref: BlockRef | None = next(
            (item for item in self.store.iter_blocks() if item.block_id == block_id),
            None,
        )
        if ref is None:
            raise FileNotFoundError(f"unknown local block: {block_id}")

        payload = ref.path.read_bytes()
        self.cache.put(block_id, payload, len(payload))
        self.prefetch.observe(block_id)
        return payload

    def status(self) -> RuntimeStatus:
        return RuntimeStatus(
            ram_budget_mb=self.config.ram_budget_mb,
            working_set_mb=self.config.working_set_mb,
            process_rss_mb=process_memory_mb(),
            resident_blocks=self.cache.stats.resident_items,
            cache_hits=self.cache.stats.hits,
            cache_misses=self.cache.stats.misses,
            offline=self.config.offline,
        )
