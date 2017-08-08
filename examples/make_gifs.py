import cv2
import numpy as np
from keras.utils import Sequence

from transform.sequences import RandomRotationTransformer, RandomHorizontalFlipTransformer, RandomShearTransformer, \
    RandomZoomTransformer

"""First, let's create a simple Sequence that load an image and resize it."""


class SimpleSequence(Sequence):
    def __init__(self, paths, shape=(200, 200)):
        self.paths = paths
        self.shape = shape
        self.batch_size = 1

    def __len__(self):
        return len(self.paths) // self.batch_size

    def __getitem__(self, index):
        paths = self.paths[index * self.batch_size:(index + 1) * self.batch_size]
        X = [cv2.resize(cv2.imread(p), self.shape) for p in paths]
        y = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in X]
        return np.array(X), np.array(y).reshape([self.batch_size, self.shape[0], self.shape[1], 1])


"""Transformers are Sequence that takes a Sequence to modify it."""
from glob import glob

paths = glob('/data/images_folder/*.jpg')
seq = SimpleSequence(paths)

"""Applying the SAME transformation to X and y is done by specifying a mask."""
transformer = RandomRotationTransformer(10, mask=[True, True])(seq)
transformer = RandomHorizontalFlipTransformer(mask=[True, True])(transformer)
transformer = RandomShearTransformer(intensity=0.5, mask=[True, True])(transformer)
transformer = RandomZoomTransformer(zoom_range=(0.8, 1.2), mask=[True, True])(transformer)

# 200,400
vid = cv2.VideoWriter(filename='/home/local/USHERBROOKE/braf3002/Documents/output.avi', fourcc=cv2.VideoWriter_fourcc(*'MJPG'), fps=10, frameSize=(400, 200),
                      isColor=True)

try:
    for i in range(100):
        X, y = transformer[0]
        im = np.concatenate((X[0], cv2.cvtColor(y[0], cv2.COLOR_GRAY2BGR)), 1)
        vid.write(im)
        cv2.imshow('Test', im)
        cv2.waitKey(100)

except:
    pass
vid.release()
