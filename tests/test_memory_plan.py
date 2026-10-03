"""Memory accounting must use real numbers and refuse plans that do not fit."""
from __future__ import annotations

import pytest

from pyrite.memory_plan import (
    MIB,
    MemoryAvailability,
    ModelFootprint,
    available_memory,
    estimate_kv_bytes,
    measure_model,
    metadata_bytes,
    plan_memory,
    require_plan,
)


class FakeTensor:
    def __init__(self, dims: tuple[int, ...], ggml_type: int = 0, size: int | None = None):
        self.dims = dims
        self.ggml_type = ggml_type
        self.size = size if size is not None else _f32_size(dims)


def _f32_size(dims: tuple[int, ...]) -> int:
    total = 4
    for dim in dims:
        total *= dim
    return total


def availability(total: int, free: int, swap: int = 0) -> MemoryAvailability:
    return MemoryAvailability(
        ram_total_bytes=total, ram_available_bytes=free, swap_total_bytes=swap, source="test"
    )


# --------------------------------------------------------------------------- #
# KV cache estimate
# --------------------------------------------------------------------------- #
def test_kv_estimate_scales_with_layers_tokens_and_width() -> None:
    base = estimate_kv_bytes(max_kv_tokens=16, num_layers=2, kv_cache_dim=8)
    doubled_layers = estimate_kv_bytes(max_kv_tokens=16, num_layers=4, kv_cache_dim=8)
    assert doubled_layers > base
    # fp32 payload: 16 tokens * 2 layers * 8 floats * 4 bytes, plus overhead.
    assert base > 16 * 2 * 8 * 4


def test_kv_estimate_of_an_empty_cache_is_zero() -> None:
    assert estimate_kv_bytes(0, 4, 8) == 0


def test_kv_estimate_rejects_negative_dimensions() -> None:
    with pytest.raises(ValueError):
        estimate_kv_bytes(-1, 4, 8)


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #
def test_available_memory_reads_the_real_operating_system() -> None:
    info = available_memory()
    assert info.known, "this environment exposes /proc/meminfo or os.sysconf"
    assert info.source in ("/proc/meminfo", "os.sysconf")
    assert info.ram_total_bytes > 0
    assert 0 < info.ram_available_bytes <= info.ram_total_bytes


def test_proc_meminfo_values_are_kilobytes_not_mebibytes() -> None:
    """Regression: the parser once multiplied kB by 1 MiB.

    On a 4 GiB machine that reported ~4 million MiB of RAM, which made every
    plan look like it fitted.
    """
    info = available_memory()
    if info.source != "/proc/meminfo":
        pytest.skip("no /proc/meminfo on this platform")
    with open("/proc/meminfo", encoding="ascii") as handle:
        meminfo_lines = handle.read().splitlines()
    raw = {}
    for line in meminfo_lines:
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            raw[key.strip()] = int(parts[0])
    assert info.ram_total_bytes == raw["MemTotal"] * 1024


# --------------------------------------------------------------------------- #
# Footprint
# --------------------------------------------------------------------------- #
def test_measure_model_sums_the_tensor_index() -> None:
    tensors = [FakeTensor((8, 8)), FakeTensor((16,)), FakeTensor((4, 4))]
    footprint = measure_model("model.gguf", tensors, largest_streamed_unit=256)
    assert footprint.tensor_bytes == sum(t.size for t in tensors)
    assert footprint.tensor_count == 3
    assert footprint.largest_tensor_bytes == 256
    # Largest leading dimension is the (16,) vector: 16 floats = 64 bytes.
    assert footprint.largest_row_bytes == 64
    assert footprint.largest_streamed_unit_bytes == 256


def test_metadata_bytes_counts_the_token_array() -> None:
    small = metadata_bytes({"general.architecture": "qwen3"})
    large = metadata_bytes({"tokenizer.ggml.tokens": ["tok"] * 1000})
    assert large > small > 0


# --------------------------------------------------------------------------- #
# Plans
# --------------------------------------------------------------------------- #
def footprint(tensor_bytes: int, largest_unit: int = 64) -> ModelFootprint:
    return ModelFootprint(
        path="model.gguf",
        file_bytes=tensor_bytes,
        tensor_bytes=tensor_bytes,
        tensor_count=1,
        largest_tensor_bytes=tensor_bytes,
        largest_row_bytes=largest_unit,
        largest_streamed_unit_bytes=largest_unit,
    )


def test_a_small_model_on_a_big_machine_plans_cleanly() -> None:
    plan = plan_memory(
        footprint(8 * MIB),
        num_layers=2,
        kv_cache_dim=16,
        max_kv_tokens=512,
        working_set_bytes=512 * MIB,
        availability=availability(8 * 1024 * MIB, 4 * 1024 * MIB),
    )
    assert plan.ok, plan.problems
    assert plan.fits_budget and plan.fits_available_ram
    assert plan.resident_requirement_bytes <= 512 * MIB
    assert plan.workspace_bytes == 512 * MIB - plan.kv_bytes


def test_the_kv_cache_eats_the_working_set_and_that_is_reported() -> None:
    plan = plan_memory(
        footprint(8 * MIB),
        num_layers=64,
        kv_cache_dim=1024,
        max_kv_tokens=131072,
        working_set_bytes=64 * MIB,
        availability=availability(8 * 1024 * MIB, 4 * 1024 * MIB),
    )
    assert not plan.ok
    assert any("KV cache" in problem for problem in plan.problems)
    with pytest.raises(MemoryError, match="PYRITE_KV_TOKENS"):
        require_plan(plan)


def test_a_resident_requirement_bigger_than_available_ram_is_refused() -> None:
    """The model itself may be huge - what must fit is the resident working set."""
    plan = plan_memory(
        footprint(64 * 1024 * MIB),
        num_layers=2,
        kv_cache_dim=16,
        max_kv_tokens=64,
        working_set_bytes=2048 * MIB,
        availability=availability(8 * 1024 * MIB, 1024 * MIB),
    )
    assert not plan.fits_available_ram
    assert any("exceeds available RAM" in problem for problem in plan.problems)


def test_a_model_bigger_than_total_ram_still_streams() -> None:
    """Larger-than-RAM is not an error: Pyrite streams it from disk."""
    plan = plan_memory(
        footprint(256 * 1024 * MIB, largest_unit=64),
        num_layers=2,
        kv_cache_dim=16,
        max_kv_tokens=64,
        working_set_bytes=512 * MIB,
        availability=availability(8 * 1024 * MIB, 6 * 1024 * MIB),
    )
    assert plan.ok, plan.problems
    assert plan.to_dict()["model_larger_than_ram"] is True
    assert any("streams it from" in note for note in plan.notes)


def test_a_streamed_unit_bigger_than_the_working_set_is_refused() -> None:
    plan = plan_memory(
        footprint(8 * MIB, largest_unit=1024 * MIB),
        num_layers=2,
        kv_cache_dim=16,
        max_kv_tokens=64,
        working_set_bytes=8 * MIB,
        availability=availability(8 * 1024 * MIB, 4 * 1024 * MIB),
    )
    assert not plan.largest_unit_fits
    assert any("largest streamed weight unit" in problem for problem in plan.problems)


def test_unknown_availability_is_reported_rather_than_assumed() -> None:
    plan = plan_memory(
        footprint(8 * MIB),
        num_layers=2,
        kv_cache_dim=16,
        max_kv_tokens=64,
        working_set_bytes=512 * MIB,
        availability=MemoryAvailability(0, 0, 0, "unknown"),
    )
    assert plan.availability.known is False
    assert plan.to_dict()["model_larger_than_ram"] is None
    assert any("no memory counters" in note for note in plan.notes)


def test_plan_dict_is_json_shaped() -> None:
    plan = plan_memory(
        footprint(8 * MIB),
        num_layers=2,
        kv_cache_dim=16,
        max_kv_tokens=64,
        working_set_bytes=512 * MIB,
        availability=availability(8 * 1024 * MIB, 4 * 1024 * MIB),
    )
    payload = plan.to_dict()
    assert payload["ok"] is True
    assert payload["streams_from_disk"] is True
    assert payload["model_mb"] == pytest.approx(8.0)
    assert set(payload) >= {"kv_cache_mb", "workspace_mb", "problems", "notes"}


def test_require_plan_passes_through_a_good_plan() -> None:
    plan = plan_memory(
        footprint(8 * MIB),
        num_layers=2,
        kv_cache_dim=16,
        max_kv_tokens=64,
        working_set_bytes=512 * MIB,
        availability=availability(8 * 1024 * MIB, 4 * 1024 * MIB),
    )
    assert require_plan(plan, what="test") is plan
