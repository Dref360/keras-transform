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


def sequential_test():
    # TODO need better test.
    sequential = SequentialTransformer([RandomZoomTransformer((0.8, 1.2)),
                                        RandomVerticalFlipTransformer()])

    _ = sequential(TestSequence())[0]
    _ = sequential(TestTreeSequence())[0]


if __name__ == '__main__':
    pytest.main([__file__])
