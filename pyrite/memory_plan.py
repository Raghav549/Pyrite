"""Honest memory accounting for loading and running a GGUF checkpoint.

Pyrite streams weights, so "the model fits" is not a single number.  Three
separate quantities decide whether a run can happen:

* **model bytes** - what the file occupies; it may exceed RAM, because tensors
  are read in bounded ranges rather than resident in full.  What must fit is the
  largest single streamed unit (one matrix row, or one expert slice).
* **KV bytes** - the compact fp32 key/value cache for the configured context.
  This *is* fully resident and grows linearly with ``max_kv_tokens``.
* **workspace bytes** - the byte-capped streaming working set, the small-tensor
  cache and transient decode buffers.

:func:`plan_memory` compares those against the memory the operating system
actually reports as available and says plainly when the answer is no.  It never
claims a model fits because the number was not checked.
"""
from __future__ import annotations

import os
import sys
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .ggml_types import row_size

MIB = 1024 * 1024


@dataclass(frozen=True)
class ModelFootprint:
    """Byte accounting for a checkpoint as it sits on disk."""

    path: str
    tensor_count: int
    tensor_bytes: int
    file_bytes: int
    largest_tensor_bytes: int
    largest_row_bytes: int
    largest_streamed_unit_bytes: int

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class MemoryAvailability:
    """Memory the operating system reports as usable right now."""

    ram_total_bytes: int
    ram_available_bytes: int
    swap_total_bytes: int
    source: str

    @property
    def known(self) -> bool:
        return self.ram_total_bytes > 0

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class MemoryPlan:
    """Whether a checkpoint can be run, and why or why not."""

    model: ModelFootprint
    kv_bytes: int
    workspace_bytes: int
    resident_requirement_bytes: int
    availability: MemoryAvailability
    budget_bytes: int
    fits_budget: bool
    fits_available_ram: bool
    largest_unit_fits: bool
    problems: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, object]:
        mib = lambda value: round(value / MIB, 1)
        return {
            "model": self.model.to_dict(),
            "model_mb": mib(self.model.tensor_bytes),
            "kv_cache_mb": mib(self.kv_bytes),
            "workspace_mb": mib(self.workspace_bytes),
            "resident_requirement_mb": mib(self.resident_requirement_bytes),
            "budget_mb": mib(self.budget_bytes),
            "ram_total_mb": mib(self.availability.ram_total_bytes),
            "ram_available_mb": mib(self.availability.ram_available_bytes),
            "availability_source": self.availability.source,
            "fits_budget": self.fits_budget,
            "fits_available_ram": self.fits_available_ram,
            "largest_streamed_unit_fits": self.largest_unit_fits,
            "model_larger_than_ram": (
                self.model.tensor_bytes > self.availability.ram_total_bytes
                if self.availability.known
                else None
            ),
            "streams_from_disk": True,
            "problems": list(self.problems),
            "notes": list(self.notes),
            "ok": self.ok,
        }


def measure_model(
    path: str | Path,
    tensors: Sequence[object],
    *,
    largest_streamed_unit: int | None = None,
) -> ModelFootprint:
    """Summarise a checkpoint's byte footprint from its parsed tensor index."""
    sizes = [int(getattr(tensor, "size", 0)) for tensor in tensors]
    rows = []
    for tensor in tensors:
        dims = tuple(getattr(tensor, "dims", ()))
        if not dims:
            continue
        try:
            rows.append(row_size(int(dims[0]), int(getattr(tensor, "ggml_type", -1))))
        except (TypeError, ValueError):
            continue
    file_bytes = 0
    try:
        file_bytes = Path(path).stat().st_size
    except OSError:
        file_bytes = sum(sizes)
    largest_unit = max(
        [value for value in (largest_streamed_unit, max(rows, default=0)) if value]
        or [0]
    )
    return ModelFootprint(
        path=str(path),
        tensor_count=len(tensors),
        tensor_bytes=sum(sizes),
        file_bytes=file_bytes,
        largest_tensor_bytes=max(sizes, default=0),
        largest_row_bytes=max(rows, default=0),
        largest_streamed_unit_bytes=largest_unit,
    )


def estimate_kv_bytes(max_kv_tokens: int, num_layers: int, kv_cache_dim: int) -> int:
    """Resident bytes for the compact fp32 KV cache, including Python overhead."""
    if min(max_kv_tokens, num_layers, kv_cache_dim) < 0:
        raise ValueError("KV dimensions must be non-negative")
    payload = max_kv_tokens * num_layers * kv_cache_dim * 4
    overhead = max_kv_tokens * num_layers * (2 * (sys.getsizeof(array("f")) + 8))
    return payload + overhead


def available_memory() -> MemoryAvailability:
    """Read real memory numbers from the operating system."""
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        values: dict[str, int] = {}
        try:
            for line in meminfo.read_text(encoding="ascii", errors="replace").splitlines():
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts and parts[0].isdigit():
                    # /proc/meminfo reports kilobytes, not kibibytes-in-MiB.
                    unit = 1024 if len(parts) > 1 and parts[1].lower() == "kb" else 1
                    values[key.strip()] = int(parts[0]) * unit
        except OSError:
            values = {}
        if values:
            total = values.get("MemTotal", 0)
            available = values.get("MemAvailable", values.get("MemFree", 0))
            return MemoryAvailability(
                ram_total_bytes=total,
                ram_available_bytes=available,
                swap_total_bytes=values.get("SwapTotal", 0),
                source="/proc/meminfo",
            )
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = os.sysconf("SC_PHYS_PAGES") * page
        available = os.sysconf("SC_AVPHYS_PAGES") * page
        return MemoryAvailability(
            ram_total_bytes=total,
            ram_available_bytes=available,
            swap_total_bytes=0,
            source="os.sysconf",
        )
    except (ValueError, OSError, AttributeError):
        return MemoryAvailability(0, 0, 0, "unknown")


def plan_memory(
    model: ModelFootprint,
    *,
    num_layers: int,
    kv_cache_dim: int,
    max_kv_tokens: int,
    working_set_bytes: int,
    availability: MemoryAvailability | None = None,
    extra_notes: Sequence[str] = (),
) -> MemoryPlan:
    """Build a :class:`MemoryPlan` and refuse to claim a fit it cannot support.

    ``working_set_bytes`` is the *total* resident budget: the KV cache is
    allocated inside it, so the streaming workspace is whatever is left over.
    """
    kv_bytes = estimate_kv_bytes(max_kv_tokens, num_layers, kv_cache_dim)
    workspace_bytes = max(0, working_set_bytes - kv_bytes)
    resident = kv_bytes + workspace_bytes
    availability = availability or available_memory()

    problems: list[str] = []
    notes: list[str] = list(extra_notes)

    if model.largest_streamed_unit_bytes > workspace_bytes:
        problems.append(
            f"the largest streamed weight unit is {model.largest_streamed_unit_bytes} bytes "
            f"but the streaming working set is only {workspace_bytes} bytes; raise the RAM "
            "budget (PYRITE_RAM_MB) or lower the reserve"
        )
    if workspace_bytes <= 0:
        problems.append(
            f"the KV cache alone needs {kv_bytes} bytes, which leaves no room in the "
            f"{working_set_bytes} byte resident budget; lower PYRITE_KV_TOKENS or raise "
            "PYRITE_RAM_MB"
        )
    if availability.known and availability.ram_available_bytes > 0:
        if resident > availability.ram_available_bytes:
            problems.append(
                f"resident requirement {resident / MIB:.1f} MiB exceeds available RAM "
                f"{availability.ram_available_bytes / MIB:.1f} MiB"
            )
        if model.largest_streamed_unit_bytes > availability.ram_available_bytes:
            problems.append(
                "even a single streamed weight unit does not fit in available RAM"
            )
        if model.tensor_bytes > availability.ram_total_bytes:
            notes.append(
                f"the checkpoint ({model.tensor_bytes / MIB:.1f} MiB) is larger than total "
                f"RAM ({availability.ram_total_bytes / MIB:.1f} MiB); Pyrite streams it from "
                "disk instead of loading it, which requires the storage to stay attached"
            )
    elif not availability.known:
        notes.append("the operating system reported no memory counters; availability is unknown")

    largest_unit_fits = model.largest_streamed_unit_bytes <= max(
        workspace_bytes, 1
    ) and (
        not availability.known
        or model.largest_streamed_unit_bytes <= max(availability.ram_available_bytes, 0)
    )
    return MemoryPlan(
        model=model,
        kv_bytes=kv_bytes,
        workspace_bytes=workspace_bytes,
        resident_requirement_bytes=resident,
        availability=availability,
        budget_bytes=working_set_bytes,
        fits_budget=workspace_bytes > 0
        and model.largest_streamed_unit_bytes <= workspace_bytes,
        fits_available_ram=not availability.known
        or (resident <= availability.ram_available_bytes),
        largest_unit_fits=largest_unit_fits,
        problems=tuple(problems),
        notes=tuple(notes),
    )


def require_plan(plan: MemoryPlan, what: str = "load") -> MemoryPlan:
    """Raise ``MemoryError`` with the concrete reasons when the plan fails."""
    if plan.ok:
        return plan
    raise MemoryError(
        f"cannot {what}: " + "; ".join(plan.problems)
        + f" (model {plan.model.tensor_bytes / MIB:.1f} MiB, "
        f"KV {plan.kv_bytes / MIB:.1f} MiB, workspace {plan.workspace_bytes / MIB:.1f} MiB, "
        f"available RAM {plan.availability.ram_available_bytes / MIB:.1f} MiB"
        f" [{plan.availability.source}])"
    )


def metadata_bytes(metadata: Mapping[str, object]) -> int:
    """Approximate the metadata payload size (dominated by the token array)."""
    total = 0
    for key, value in metadata.items():
        total += len(key) + 8
        if isinstance(value, (bytes, bytearray)):
            total += len(value)
        elif isinstance(value, (list, tuple)):
            total += sum(len(item) if isinstance(item, str) else 8 for item in value)
        elif isinstance(value, str):
            total += len(value)
        else:
            total += 8
    return total
