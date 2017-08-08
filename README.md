# keras-transform
Library for data augmentation

This library will provide a data augmentation pipeline for `Sequence` objects.

**Keras-transform** allows the user to specify a mask to do data augmentation on the groundtruth or any way you would like.
See [simple.ipynb](examples/simple.ipynb).

![alt-text](/examples/example.gif)

# Contributing
Anyone can contribute by submitting a PR, but remember that the code is in an early stage so things may change.
Also, a lot of documentation needs to be done.
Any PR that adds a new feature needs to be tested.

Here's an example where X is an image and the groundtruth is the grayscale. The code can be found [here](examples/make_gifs.py)


# TODO
- [x] Handle masked augmentation
- [x] Random rotation
- [x] CircleCI
- [x] Random zoom
- [x] Random shift
- [x] Random channel shift
- [x] Random shear
- [x] Random flip
- [ ] Functional API utils


