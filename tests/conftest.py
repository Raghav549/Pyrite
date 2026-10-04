"""Suite-wide fixtures.

The memory gate in :func:`pyrite.memory_plan.require_plan` compares the
configured resident budget against *currently available* RAM.  The default
budget is 4096 MiB minus the 768 MiB reserve, i.e. 3328 MiB resident - more
than this machine often has available once other tests have run.  Left alone,
any test that opens an executor fails with ``MemoryError`` depending on nothing
but whatever else happens to be resident, which makes the suite
non-deterministic.

Pinning ``PYRITE_RAM_MB`` for every test removes that dependency.  Tests that
specifically want to observe the shipped default opt back out with
``monkeypatch.delenv("PYRITE_RAM_MB", raising=False)`` - see ``tests/test_cli.py``.
"""
from __future__ import annotations

import pytest

#: 2048 MiB budget - 768 MiB reserve = 1280 MiB of streaming working set:
#: comfortably above the largest streamed unit in any fixture, and comfortably
#: below the RAM available even on a busy small machine.
TEST_RAM_BUDGET_MB = "2048"


@pytest.fixture(autouse=True)
def _pinned_ram_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYRITE_RAM_MB", TEST_RAM_BUDGET_MB)
