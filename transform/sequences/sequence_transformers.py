import keras.backend as K
import numpy as np
from keras.utils import Sequence

from transform.utils import apply_fun, get_batch_shape
from transform.utils.transformations import random_rotation, random_shift


class BaseSequenceTransformer(Sequence):
    """Base object for transformers.

    # Arguments
        sequence: Sequence object to iterate over.
        data_format: `'channels_last'`, `'channels_first'` or None
        mask: A tree-like boolean structure to know which data to transform.
    """

    def on_epoch_end(self):
        pass

    def __init__(self, sequence, data_format=None, mask=True):
        self.sequence = sequence
        self.mask = mask
        self.batch_size = None  # We do not know yet
        self.transformation = id
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

    def apply_transformation(self, x_, transformation, args):
        return np.asarray(
            list(map(lambda args: transformation(args[0], row_axis=self.row_axis, col_axis=self.col_axis,
                                                 channel_axis=self.channel_axis - 1, **args[1]),
                     zip(x_, args))))

    def get_args(self):
        """Retrieve args to provide to the transformer. The args should not be aware of the input dimension.

        # Returns
            A list of batch_size args.
        """
        raise NotImplementedError

    def __getitem__(self, index):
        batch = self.sequence[index]
        if self.batch_size is None:
            # The first batch should be the maximum batch_size i.e. not the last.
            self.batch_size = get_batch_shape(batch)[0]

        args = self.get_args()

        return apply_fun(batch, self.apply_transformation, self.mask, transformation=self.transformation, args=args)

    def __len__(self):
        return len(self.sequence)


class RandomRotationTransformer(BaseSequenceTransformer):
    """Transformer to do random rotation.

    # Arguments
        sequence: Sequence object to iterate over.
        rg: Range of rotation
    """

    def __init__(self, sequence, rg, mask=(True, False)):
        super().__init__(sequence, mask=mask)
        self.rg = rg
        self.transformation = random_rotation

    def get_args(self):
        return [{'rg': self.rg,
                 'theta': np.pi / 180 * np.random.uniform(-self.rg, self.rg)} for _ in range(self.batch_size)]


class RandomShiftTransformer(BaseSequenceTransformer):
    """Transformer to do random shift.

    # Arguments
        sequence: Sequence object to iterate over.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
    """

    def __init__(self, sequence, wrg, hrg, mask=(True, False)):
        super().__init__(sequence, mask=mask)
        self.wrg = wrg
        self.hrg = hrg
        self.transformation = random_shift

    def get_args(self):
        tx = np.random.uniform(-self.hrg, self.hrg)
        ty = np.random.uniform(-self.wrg, self.wrg)
        return [{'tx': tx, 'ty': ty, 'wrg': self.wrg, 'hrg': self.hrg} for _ in range(self.batch_size)]
