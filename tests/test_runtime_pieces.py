"""Coverage for the remaining runtime pieces: policy, scheduler, storage pages.

These modules back the memory-budget story, so each test asserts behaviour that
is part of the promise rather than an implementation detail.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pyrite.execution_loop import FeedbackExecutionLoop
from pyrite.gpu import detect_accelerators
from pyrite.io_scheduler import IOComputeTask
from pyrite.model_loader import LocalTensorLoader
from pyrite.policy import DevicePolicy
from pyrite.privacy import PrivacyPolicy
from pyrite.scheduler import BlockScheduler
from pyrite.storage_pages import ContentAddressedPager, MMapPage
from pyrite.weight_page import PageManifest

from .tiny_qwen3moe import build_tiny_checkpoint


# ------------------------------------------------------------------- policy
def test_device_policy_rejects_an_impossible_budget():
    with pytest.raises(ValueError):
        DevicePolicy(ram_budget_mb=512, working_set_mb=256).validate()
    with pytest.raises(ValueError):
        DevicePolicy(ram_budget_mb=4096, working_set_mb=0).validate()


def test_device_policy_rejects_offline_with_network_features():
    policy = DevicePolicy(
        ram_budget_mb=4096,
        working_set_mb=1024,
        privacy=PrivacyPolicy(offline_only=True, telemetry=True),
    )
    with pytest.raises(ValueError):
        policy.validate()


def test_privacy_policy_blocks_network_when_offline():
    with pytest.raises(PermissionError):
        PrivacyPolicy(offline_only=True).assert_network_allowed()
    PrivacyPolicy(offline_only=False).assert_network_allowed()


# ---------------------------------------------------------------- scheduler
def test_block_scheduler_plans_required_prefetch_and_evictions():
    scheduler = BlockScheduler(resident_limit=2, prefetch_depth=2)
    blocks = ["a", "b", "c", "d"]
    plan = scheduler.plan(blocks, 1, resident={"a", "z"})
    assert plan.required == ("b",)
    assert plan.prefetch == ("c", "d")
    assert plan.evict == ("a", "z")


def test_block_scheduler_handles_the_last_block():
    scheduler = BlockScheduler(prefetch_depth=3)
    plan = scheduler.plan(["only"], 0, resident=set())
    assert plan.required == ("only",)
    assert plan.prefetch == ()


# --------------------------------------------------------------------- gpu
def test_accelerator_detection_reports_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    by_name = {info.name: info for info in detect_accelerators()}
    assert by_name["cuda"].available is True
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert {info.name: info for info in detect_accelerators()}["cuda"].available is False


def test_accelerator_detection_always_reports_its_reason(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    for info in detect_accelerators():
        assert info.reason
        assert info.name in {"cuda", "metal"}


# --------------------------------------------------------------- storage
def test_page_manifest_covers_the_file(tmp_path: Path):
    payload = bytes(range(256)) * 40  # 10 KiB
    source = tmp_path / "weights.bin"
    source.write_bytes(payload)

    refs = PageManifest(page_bytes=4096).build(source)
    assert len(refs) == 3
    assert sum(ref.size for ref in refs) == len(payload)
    assert [ref.offset for ref in refs] == [0, 4096, 8192]
    for ref in refs:
        assert ref.digest == __import__("hashlib").sha256(payload[ref.offset:ref.offset + ref.size]).hexdigest()


def test_content_addressed_pager_deduplicates_identical_pages(tmp_path: Path):
    page = b"x" * 1024
    source = tmp_path / "weights.bin"
    source.write_bytes(page * 3)
    pager = ContentAddressedPager(tmp_path / "pages", page_bytes=1024)
    refs = pager.ingest(source)
    assert len(refs) == 3
    assert len({ref.page_id for ref in refs}) == 1
    assert len(list((tmp_path / "pages").glob("*.page"))) == 1


def test_mmap_page_view_matches_the_file(tmp_path: Path):
    source = tmp_path / "weights.bin"
    source.write_bytes(b"abcdef" * 100)
    pager = ContentAddressedPager(tmp_path / "pages", page_bytes=64)
    first = pager.ingest(source)[0]
    with MMapPage(first) as view:
        mapped = view.view()
        assert bytes(mapped) == (b"abcdef" * 100)[:64]
        assert len(mapped) == first.size


# ------------------------------------------------------- tensor loader
def test_tensor_loader_reads_a_real_gguf(tmp_path: Path):
    path, _cfg, tensors = build_tiny_checkpoint(tmp_path / "tiny.gguf")
    with LocalTensorLoader(path) as loader:
        names = {info.name for info in loader.tensors()}
        assert "token_embd.weight" in names
        assert len(names) == 27
        raw = loader.load("output_norm.weight")
        assert raw == tensors["output_norm.weight"]
        assert loader.name_for("tensor:blk.0.attn_q.weight") == "blk.0.attn_q.weight"
        with pytest.raises(KeyError):
            loader.load("does.not.exist")


# ----------------------------------------------------------------- loop
def test_feedback_loop_records_measurements_in_plan_order():
    loop = FeedbackExecutionLoop()
    tasks = [
        IOComputeTask("second", io_cost_bytes=2048, compute_cost=1.0, priority=0.0),
        IOComputeTask("first", io_cost_bytes=64, compute_cost=1.0, priority=10.0),
    ]
    order = []

    def load(task_id: str) -> str:
        order.append(f"load:{task_id}")
        return task_id.upper()

    def compute(task_id: str, value: object) -> str:
        order.append(f"compute:{task_id}")
        return f"{value}!"

    results = loop.run("plan-a", tasks, load, compute)
    assert [result.task_id for result in results] == ["first", "second"]
    assert results[0].output == "FIRST!"
    assert order == ["load:first", "compute:first", "load:second", "compute:second"]
    stats = loop.feedback.stats["plan-a"]
    assert stats.executions == 2
    assert stats.total_io_bytes == 64 + 2048
    assert loop.scheduler.ema_io_seconds >= 0.0


def test_mmap_page_rejects_an_empty_page(tmp_path: Path):
    from pyrite.storage_pages import WeightPage

    page = WeightPage("empty", tmp_path / "empty.page", 0, 0, "x" * 64)
    with pytest.raises(ValueError, match="empty page"):
        MMapPage(page)
