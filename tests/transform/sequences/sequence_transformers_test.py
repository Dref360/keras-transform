import numpy as np
import pytest
from keras.utils import Sequence

from transform.sequences import RandomRotationTransformer


class TestSequence(Sequence):
    def __getitem__(self, index):
        return np.arange(5*20*20*3).reshape([5, 20, 20, 3]), np.arange(5*20*20*3).reshape([5, 20, 20, 3])

    def __len__(self):
        return 10


class TestTreeSequence(Sequence):
    def __getitem__(self, index):
        return [np.arange(5*20*20*3).reshape([5, 20, 20, 3]), np.arange(5*12*12*3).reshape([5, 12, 12, 3])], np.arange(5*10*10*3).reshape([5, 10, 10, 3])

    def __len__(self):
        return 10


def test_random_rot():
    np.random.seed(1337)
    transformer = RandomRotationTransformer(TestSequence(), 25)
    assert np.any(np.not_equal(transformer[0][0], transformer[1][0])) and np.all(
        np.equal(transformer[0][1], transformer[1][1]))

    transformer = RandomRotationTransformer(TestTreeSequence(), 25)

    assert all([np.any(np.not_equal(t0, t1)) for t0, t1 in zip(transformer[0][0], transformer[1][0])]) and all(
        [np.all(np.equal(t0, t1)) for t0, t1 in zip(transformer[0][1], transformer[1][1])])

    # Test Mask
    transformer = RandomRotationTransformer(TestTreeSequence(), 25, mask=False)

    assert all([np.any(np.equal(t0, t1)) for t0, t1 in zip(transformer[0][0], transformer[1][0])]) and np.equal(
        transformer[0][1], transformer[1][1]).all()

    transformer = RandomRotationTransformer(TestTreeSequence(), 25, mask=[True, True])

    assert all([np.any(np.not_equal(t0, t1)) for t0, t1 in zip(transformer[0][0], transformer[1][0])]) and np.not_equal(
        transformer[0][1], transformer[1][1]).any()

    # Should rotate the same way for X and y
    transformer = RandomRotationTransformer(TestSequence(), 25, mask=[True, True])
    assert (np.equal(*transformer[0])).all()

    # Common case where we augment X but not y
    transformer = RandomRotationTransformer(TestSequence(), 25, mask=[True, False])
    assert (np.not_equal(*transformer[0])).any()


if __name__ == '__main__':
    pytest.main([__file__])
