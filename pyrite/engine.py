from __future__ import annotations

from dataclasses import dataclass

from .config import RuntimeConfig
from .gpu import AcceleratorInfo, detect_accelerators
from .kernels.t_sar import ReferenceTernaryKernel
from .kv import KVBudget
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
    resident_bytes: int
    cache_hits: int
    cache_misses: int
    offline: bool

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


class PyriteRuntime:
    """Owns the memory budget, block store, cache and scheduling policy."""

    def __init__(self, config: RuntimeConfig | None = None):
        self.config = config or RuntimeConfig.from_env()
        self.config.validate()
        self.config.storage_dir.mkdir(parents=True, exist_ok=True)

        self.policy = DevicePolicy(
            self.config.ram_budget_mb,
            self.config.working_set_mb,
        )
        self.policy.validate()

        self.store = LocalBlockStore(self.config.storage_dir)
        self.cache = LRUResidentCache[bytes](
            self.config.resident_blocks,
            self.config.resident_byte_budget,
        )
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

    def accelerators(self) -> tuple[AcceleratorInfo, ...]:
        return tuple(detect_accelerators())

    def plan(self, blocks: list[str], current_index: int = 0) -> LoadPlan:
        if not blocks:
            return LoadPlan((), (), ())
        index = current_index if 0 <= current_index < len(blocks) else 0
        current = blocks[index]
        candidates = blocks[index + 1:]
        predicted = self.prefetch.predict(
            current,
            candidates,
            self.config.prefetch_depth,
        )
        plan = self.scheduler.plan(blocks, index, set(self.cache.keys()))
        return LoadPlan(plan.required, predicted or plan.prefetch, plan.evict)

    def load_block_bytes(self, block_id: str) -> bytes:
        hit = self.cache.get(block_id)
        if hit is not None:
            self.prefetch.observe(block_id)
            return hit

        ref: BlockRef | None = self.store.get(block_id)
        if ref is None:
            raise FileNotFoundError(f"unknown local block: {block_id}")

        if ref.size_bytes > self.config.resident_byte_budget:
            raise MemoryError(
                f"block {block_id!r} is larger than the configured resident budget"
            )

        payload = ref.path.read_bytes()
        self.cache.put(block_id, payload, len(payload))
        self.prefetch.observe(block_id)
        return payload

    def status(self) -> RuntimeStatus:
        return RuntimeStatus(
            self.config.ram_budget_mb,
            self.config.working_set_mb,
            process_memory_mb(),
            self.cache.stats.resident_items,
            self.cache.stats.estimated_bytes,
            self.cache.stats.hits,
            self.cache.stats.misses,
            self.config.offline,
        )

    def describe(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "status": self.status().to_dict(),
            "accelerators": [info.__dict__ for info in self.accelerators()],
            "kv": {
                "max_tokens": self.kv.max_tokens,
                "tokens_seen": self.kv.stats.tokens_seen,
                "tokens_kept": self.kv.stats.tokens_kept,
                "tokens_dropped": self.kv.stats.tokens_dropped,
            },
        }
