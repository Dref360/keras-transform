import keras.backend as K
import numpy as np
from keras.utils import Sequence

from transform.utils import apply_fun, get_batch_size
from transform.utils.transformations import (random_rotation, random_shift, random_zoom, random_channel_shift,
                                             random_shear, flip_horizontal, flip_vertical)


class BaseSequenceTransformer(Sequence):
    """Base object for transformers.

    # Arguments
        data_format: `'channels_last'`, `'channels_first'` or None
    """

    def __init__(self, data_format=None):
        self.sequence = None
        self.mask = None
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

        self.common_args = {'row_axis': self.row_axis, 'col_axis': self.col_axis,
                            'channel_axis': self.channel_axis - 1}

    def __call__(self, seq, mask=(True, False)):
        self.mask = mask
        self.sequence = seq
        return self

    def on_epoch_end(self):
        pass

    def apply_transformation(self, x_, transformation, args):
        """
        Apply the `transformation` to the input `x_`.
        :param x_: np.array, the input
        :param transformation: function to apply
        :param args: dict, arguments for `transformation`
        :return: np.array
        """
        return np.asarray(
            list(map(lambda args: transformation(args[0], **args[1]),
                     zip(x_, args))))

    def get_args(self):
        """Retrieve args to provide to the transformer. The args should not be aware of the input dimension.

        # Returns
            A list of batch_size args.
        """
        raise NotImplementedError

    def __getitem__(self, index):
        assert self.sequence, "This transformer {} has not been called with a Sequence object".format(
            self.__class__.__name__)
        batch = self.sequence[index]
        if self.batch_size is None:
            # The first batch should be the maximum batch_size i.e. not the last.
            self.batch_size = get_batch_size(batch)

        args = self.get_args()
        for arg in args:
            arg.update(self.common_args)

        return apply_fun(batch, self.apply_transformation, self.mask, transformation=self.transformation, args=args)

    def __len__(self):
        return len(self.sequence)


class RandomRotationTransformer(BaseSequenceTransformer):
    """Transformer to do random rotation.

    # Arguments
        rg: Range of rotation
    """

    def __init__(self, rg):
        super().__init__()
        self.rg = rg
        self.transformation = random_rotation

    def get_args(self):
        return [{'rg': self.rg,
                 'theta': np.pi / 180 * np.random.uniform(-self.rg, self.rg)} for _ in range(self.batch_size)]


class RandomShiftTransformer(BaseSequenceTransformer):
    """Transformer to do random shift.

    # Arguments
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
    """

    def __init__(self, wrg, hrg):
        super().__init__()
        self.wrg = wrg
        self.hrg = hrg
        self.transformation = random_shift

    def get_args(self):
        return [{'tx': np.random.uniform(-self.hrg, self.hrg), 'ty': np.random.uniform(-self.wrg, self.wrg),
                 'wrg': self.wrg, 'hrg': self.hrg} for _ in range(self.batch_size)]


class RandomZoomTransformer(BaseSequenceTransformer):
    """Transformer to do random zoom.

    # Arguments
        zoom_range: Tuple of floats; zoom range for width and height.
    """

    def __init__(self, zoom_range):
        super().__init__()
        self.zoom_range = zoom_range
        self.transformation = random_zoom

    def get_args(self):
        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            dt = [(1, 1) for _ in range(self.batch_size)]
        else:
            dt = [np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2) for _ in range(self.batch_size)]
        return [{'z_known': d, 'zoom_range': self.zoom_range} for d in dt]


class RandomChannelShiftTransformer(BaseSequenceTransformer):
    """Transformer to do random zoom.

    # Arguments
        intensity: float, intensity range
    """

    def __init__(self, intensity):
        super().__init__()
        self.intensity = intensity
        self.transformation = random_channel_shift
        self.common_args = {'channel_axis': self.channel_axis - 1}

    def get_args(self):
        return [{'known_intensity': np.random.uniform(-self.intensity, self.intensity), 'intensity': self.intensity} for
                _ in range(self.batch_size)]


class RandomShearTransformer(BaseSequenceTransformer):
    """Transformer to do random shear.

    # Arguments
        intensity: float, maximum shear.
    """

    def __init__(self, intensity):
        super().__init__()
        self.intensity = intensity
        self.transformation = random_shear

    def get_args(self):
        return [{'known_intensity': np.random.uniform(-self.intensity, self.intensity), 'intensity': self.intensity} for
                _ in range(self.batch_size)]


class RandomHorizontalFlipTransformer(BaseSequenceTransformer):
    """Transformer to do random horizontal flip."""
    def __init__(self):
        super().__init__()
        self.transformation = flip_horizontal
        # The -1 is important here!
        self.common_args = {'col_axis': self.col_axis - 1}

    def get_args(self):
        return [{'value': np.random.random()} for
                _ in range(self.batch_size)]


class RandomVerticalFlipTransformer(BaseSequenceTransformer):
    """Transformer to do random vertical flip."""
    def __init__(self):
        super().__init__()
        self.transformation = flip_vertical
        # The -1 is important here!
        self.common_args = {'row_axis': self.row_axis - 1}

    def get_args(self):
        return [{'value': np.random.random()} for
                _ in range(self.batch_size)]
