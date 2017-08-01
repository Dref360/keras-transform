from setuptools import setup
from setuptools import find_packages


setup(name='keras-transform',
      version='0.0.1',
      description='Library for data augmentation',
      author='Frédéric Branchaud-Charron',
      author_email='frederic.branchaud-charron@usherbrooke.ca',
      url='https://github.com/Dref360/keras-transform',
      license='MIT',
      install_requires=['keras>=2.0.5'],
      extras_require={
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'pytest-cov'],
      },
      packages=find_packages())
