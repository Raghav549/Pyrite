from pyrite.tensor import DenseTensor, matvec


def test_tensor_matvec():
    t = DenseTensor(2, 2, (1.0, 2.0, 3.0, 4.0))
    assert matvec(t, (5.0, 6.0)) == (17.0, 39.0)
