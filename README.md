# keras-transform
Library for data augmentation

This library provides a data augmentation pipeline for `Sequence` objects.

**Keras-transform** allows the user to specify a mask to do data augmentation in a flexible way. This is useful in many tasks like segmentation where we want the ground truth to be augmented.
See [simple.ipynb](examples/simple.ipynb).

**Keras-transform** also works with multiple inputs, outputs by using complex masks.
For example, `mask=[[True,False],False]` would augment the first input but not the second.

## keras-transform in 10 lines

```python
from transform.sequences import SequentialTransformer
from transform.sequences import RandomZoomTransformer, RandomVerticalFlipTransformer

seq = ... # A keras.utils.Sequence object that returns (X,y)
model = ... # A keras Model

sequence = SequentialTransformer([RandomZoomTransformer(zoom_range=(0.8,1.2)), RandomVerticalFlipTransformer()])

# To augment X but not y
augmented_sequence = sequence(seq,mask=[True,False])
model.fit_generator(augmented_sequence,steps_per_epoch=len(augmented_sequence))

# To augment X and y
augmented_sequence = sequence(seq,mask=[True,True]) # Alternatively, mask=True would also work.
model.fit_generator(augmented_sequence,steps_per_epoch=len(augmented_sequence))

```



# Contributing
Anyone can contribute by submitting a PR.
Any PR that adds a new feature needs to be tested.

# Example

Here's an example where X is an image and the ground truth is the grayscale version of the input. The code can be found [here](examples/make_gifs.py).

![alt-text](/examples/example.gif)


