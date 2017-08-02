import keras.backend as K
import numpy as np
from transform.utils.transformations import random_rotation
from transform.utils import apply_fun
from keras.utils import Sequence


class BaseSequenceTransformer(Sequence):
    """Base object for transformers.

    # Arguments
        sequence: Sequence object to iterate over.
        data_format: `'channels_last'`, `'channels_first'` or None
        mask: A tree-like boolean structure to know which data to transform.
    """
    def __init__(self, sequence, data_format=None, mask=True):
        self.sequence = sequence
        self.mask = mask
        if data_format is None:
            data_format = K.image_data_format()
        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('`data_format` should be `"channels_last"` (channel after row and '
                             'column) or `"channels_first"` (channel before row and column). '
                             'Received arg: ', data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2

    def __getitem__(self, index):
        return self.sequence[index]

    def __len__(self):
        return len(self.sequence)


class RandomRotationTransformer(BaseSequenceTransformer):
    """Transformer to do random rotation.

    # Arguments
        sequence: Sequence object to iterate over.
        rg: Range of rotation
    """
    def __init__(self, sequence, rg,mask=(True,False)):
        super().__init__(sequence,mask=mask)
        self.rg = rg

    def _rotate_batch(self, x_, theta=None):
        return np.asarray(list(map(lambda l: random_rotation(l, rg=self.rg, row_axis=self.row_axis, col_axis=self.col_axis,
                                             channel_axis=self.channel_axis-1, theta=theta), x_)))

    def __getitem__(self, index):
        batch = self.sequence[index]
        theta = np.pi / 180 * np.random.uniform(-self.rg, self.rg)

        return apply_fun(batch,self._rotate_batch,self.mask,theta=theta)
