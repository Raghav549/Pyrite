from pyrite.compute import ComputeBackend


def test_reference_matvec():
    backend = ComputeBackend()
    out = backend.matvec_f32([[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0])
    assert out == [17.0, 39.0]
