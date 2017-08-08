import numpy as np
import pytest
from keras.utils import Sequence

from transform.sequences import (RandomRotationTransformer, RandomShiftTransformer, RandomZoomTransformer,
                                 RandomChannelShiftTransformer, RandomShearTransformer, RandomHorizontalFlipTransformer,
                                 RandomVerticalFlipTransformer
                                 )


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


def test_random_rot():
    np.random.seed(1337)
    inner_transformer(RandomRotationTransformer, rg=25)


def test_random_shift():
    np.random.seed(1337)
    inner_transformer(RandomShiftTransformer, wrg=0.5, hrg=0.5)


def test_random_zoom():
    np.random.seed(1337)
    inner_transformer(RandomZoomTransformer, zoom_range=(.2, 1.5))


def test_random_intensity_shift():
    np.random.seed(1337)
    inner_transformer(RandomChannelShiftTransformer, intensity=10)


def test_random_shear():
    np.random.seed(1337)
    inner_transformer(RandomShearTransformer, intensity=10)


def test_random_flip():
    np.random.seed(1337)
    # This SHOULD work since batch_size is 5 and we have 50% chances of doing a flip.
    inner_transformer(RandomHorizontalFlipTransformer)
    inner_transformer(RandomVerticalFlipTransformer)


def test_assert():
    with pytest.raises(AssertionError):
        _ = RandomHorizontalFlipTransformer()[0]


def inner_transformer(transformer_cls, **kwargs):
    transformer = transformer_cls(**kwargs)(TestSequence())
    # Assert that X changes between 2 calls and Y does not.
    assert np.any(np.not_equal(transformer[0][0], transformer[1][0])) and np.all(
        np.equal(transformer[0][1], transformer[1][1]))

    transformer = transformer_cls(**kwargs)(TestTreeSequence())

    assert all([np.any(np.not_equal(t0, t1)) for t0, t1 in zip(transformer[0][0], transformer[1][0])]) and all(
        [np.all(np.equal(t0, t1)) for t0, t1 in zip(transformer[0][1], transformer[1][1])])

    # Test Mask
    transformer = transformer_cls(**kwargs)(TestTreeSequence(), mask=False)

    assert all([np.any(np.equal(t0, t1)) for t0, t1 in zip(transformer[0][0], transformer[1][0])]) and np.equal(
        transformer[0][1], transformer[1][1]).all()

    transformer = transformer_cls(**kwargs)(TestTreeSequence(), mask=[True, True])

    assert all([np.any(np.not_equal(t0, t1)) for t0, t1 in zip(transformer[0][0], transformer[1][0])]) and np.not_equal(
        transformer[0][1], transformer[1][1]).any()

    # Should transform the same way for X and y
    transformer = transformer_cls(**kwargs)(TestSequence(), mask=[True, True])
    assert (np.equal(*transformer[0])).all()

    # Common case where we augment X but not y
    transformer = transformer_cls(**kwargs)(TestSequence(), mask=[True, False])
    assert (np.not_equal(*transformer[0])).any()


if __name__ == '__main__':
    pytest.main([__file__])
