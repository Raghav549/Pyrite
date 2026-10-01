from pyrite.config import RuntimeConfig
from pyrite.engine import PyriteRuntime
from pyrite.kernels.t_sar import ReferenceTernaryKernel


def test_four_gb_budget_defaults():
    rt = PyriteRuntime(RuntimeConfig(ram_budget_mb=4096, reserve_mb=512))
    status = rt.status()
    assert status.ram_budget_mb == 4096
    assert status.working_set_mb == 3584
    assert status.offline is True


def test_router():
    rt = PyriteRuntime(RuntimeConfig())
    assert rt.route("write Python code for an API").name == "coding"


def test_kv_budget():
    rt = PyriteRuntime(RuntimeConfig(max_kv_tokens=10))
    assert rt.kv.accept(7) == 7
    assert rt.kv.accept(7) == 3
    assert rt.kv.stats.tokens_dropped == 4


def test_ternary_reference_kernel():
    kernel = ReferenceTernaryKernel()
    assert kernel.matvec([[1, 0, -1], [-1, 1, 0]], [2, 3, 4]) == [-2, 1]
    assert kernel.estimate_bits(2, 3) == 12
