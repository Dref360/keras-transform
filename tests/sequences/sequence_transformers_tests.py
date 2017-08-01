import numpy as np
from keras.utils import Sequence

from transform.sequences import RandomRotationTransformer


class TestSequence(Sequence):
    def __getitem__(self, index):
        return np.ones([5, 20, 20, 3]), 3

    def __len__(self):
        return 10


class TestTreeSequence(Sequence):
    def __getitem__(self, index):
        return [np.ones([5, 20, 20, 3]), np.ones([5, 12, 12, 3])], 3

    def __len__(self):
        return 10


def test_random_rot():
    np.random.seed(1337)
    transformer = RandomRotationTransformer(TestSequence(), 25)
    assert np.any(np.not_equal(transformer[0][0], transformer[1][0]))

    transformer = RandomRotationTransformer(TestTreeSequence(), 25)

    assert all([np.any(np.not_equal(t0,t1)) for t0, t1 in zip(transformer[0][0], transformer[1][0])])
