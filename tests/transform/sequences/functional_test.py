import numpy as np
import pytest
from keras.utils import Sequence

from transform.sequences import SequentialTransformer, RandomZoomTransformer, RandomVerticalFlipTransformer


class TestSequence(Sequence):
    """Create a X,Y tuple"""

    def __getitem__(self, index):
        return np.arange(5 * 20 * 20 * 3).reshape([5, 20, 20, 3]), np.arange(5 * 20 * 20 * 3).reshape([5, 20, 20, 3])

    def __len__(self):
        return 10


class TestTreeSequence(Sequence):
    """Create a [X1,X2],Y1 tuple."""

    def __getitem__(self, index):
        return [np.arange(5 * 20 * 20 * 3).reshape([5, 20, 20, 3]),
                np.arange(5 * 12 * 12 * 3).reshape([5, 12, 12, 3])], np.arange(
            5 * 10 * 10 * 3).reshape([5, 10, 10, 3])

    def __len__(self):
        return 10


def inner_transformer(transformer_obj, **kwargs):
    transformer = transformer_obj(TestSequence())
    # Assert that X changes between 2 calls and Y does not.
    assert np.any(np.not_equal(transformer[0][0], transformer[1][0])) and np.all(
        np.equal(transformer[0][1], transformer[1][1]))

    transformer = transformer_obj(TestTreeSequence())

    assert all([np.any(np.not_equal(t0, t1)) for t0, t1 in zip(transformer[0][0], transformer[1][0])]) and all(
        [np.all(np.equal(t0, t1)) for t0, t1 in zip(transformer[0][1], transformer[1][1])])

    # Test Mask
    transformer = transformer_obj(TestTreeSequence(), mask=False)

    assert all([np.any(np.equal(t0, t1)) for t0, t1 in zip(transformer[0][0], transformer[1][0])]) and np.equal(
        transformer[0][1], transformer[1][1]).all()

    transformer = transformer_obj(TestTreeSequence(), mask=[True, True])

    assert all(
        [np.any(np.not_equal(t0, t1)) for t0, t1 in zip(transformer[0][0], transformer[1][0])]) and np.not_equal(
        transformer[0][1], transformer[1][1]).any()

    # Should transform the same way for X and y
    transformer = transformer_obj(TestSequence(), mask=[True, True])
    assert (np.equal(*transformer[0])).all()

    # Common case where we augment X but not y
    transformer = transformer_obj(TestSequence(), mask=[True, False])
    assert (np.not_equal(*transformer[0])).any()


def test_sequential():
    # TODO need better test.
    sequential = SequentialTransformer([RandomZoomTransformer((0.8, 1.2)),
                                        RandomVerticalFlipTransformer()])

    inner_transformer(sequential)


if __name__ == '__main__':
    pytest.main([__file__])
