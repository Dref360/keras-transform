import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from transform.utils import get_value, handle_mask, apply_fun


def is_same(arr1, arr2):
    if isinstance(arr1, (list, tuple)):
        return all([is_same(a1, a2) for a1, a2 in zip(arr1, arr2)])
    return np.allclose(arr1, arr2)


def test_get_value():
    # Always true
    tree = True
    assert get_value(tree, [0]) is True
    tree = [True]
    assert get_value(tree, [0]) is True
    # Common pattern where x is transformed, not y
    tree = [True, False]
    assert get_value(tree, [0]) is True
    assert get_value(tree, [1]) is False

    # Mixed inputs
    tree = [[True, False], False]
    assert get_value(tree, [0, 0]) is True
    assert get_value(tree, [0, 1]) is False
    assert get_value(tree, [1]) is False

    # Mixed output
    tree = [[True, False], [False, True]]
    assert get_value(tree, [1, 0]) is False
    assert get_value(tree, [1, 1]) is True


def test_handle_mask():
    # Always true
    mask = True
    assert handle_mask(mask, [0]) == [True]
    mask = [True]
    assert handle_mask(mask, [0]) == [True]
    # Common pattern where x is transformed, not y
    mask = [True, False]
    assert handle_mask(mask, [0, 1]) == [True, False]


def test_apply_fun():
    def fun(x):
        return x * 0.0

    inp = np.ones([10])
    out = np.zeros([10])

    assert_almost_equal(apply_fun(inp.copy(), fun, True), out)
    assert_almost_equal(apply_fun(inp.copy(), fun, False), inp)

    assert is_same(apply_fun([inp.copy()], fun, True), [out])

    assert is_same(apply_fun([inp.copy()], fun, True), [out])
    assert is_same(apply_fun([inp.copy()], fun, False), [inp])

    assert is_same(apply_fun([inp.copy(), inp.copy()], fun, True), [out, out])
    assert is_same(apply_fun([inp.copy(), inp.copy()], fun, False), [inp, inp])

    assert is_same(apply_fun([inp.copy(), inp.copy()], fun, [True, False]), [out, inp])
    assert is_same(apply_fun([inp.copy(), inp.copy()], fun, [False, True]), [inp, out])

    assert is_same(apply_fun([[inp.copy(), inp.copy()], inp.copy()], fun, [True, False]), [[out, out], inp])
    assert is_same(apply_fun([[inp.copy(), inp.copy()], inp.copy()], fun, [False, True]), [[inp, inp], out])

    assert is_same(apply_fun([[inp.copy(), inp.copy()], inp.copy()], fun, [[True, True], False]), [[out, out], inp])
    assert is_same(apply_fun([[inp.copy(), inp.copy()], inp.copy()], fun, [[False, True], True]), [[inp, out], out])

    assert is_same(apply_fun([[inp.copy(), inp.copy()], inp.copy()], fun, [[False, True], False]), [[inp, out], inp])


if __name__ == '__main__':
    pytest.main([__file__])
