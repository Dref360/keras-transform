from setuptools import setup
from setuptools import find_packages

setup(name='keras-transform',
      version='0.1.1',
      description='Library for data augmentation',
      author='Frederic Branchaud-Charron',
      author_email='frederic.branchaud-charron@usherbrooke.ca',
      url='https://github.com/Dref360/keras-transform',
      license='MIT',
      install_requires=['numpy', 'theano', 'keras>=2.2.0'],
      extras_require={
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'pytest-cov'],
      },
      packages=find_packages())
