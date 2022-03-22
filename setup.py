from setuptools import setup
from distutils.command import build as build_module

setup(
  name='poetryT5',
  version='1.0',
  description='Poetry Generation',
  author='Sebastian Ochs, Julien Brosseit, Cleo Matzken, The-Khang Nguyen',
  packages=['poetryT5'],
  entry_points={
      'console_scripts': [
        'pt5-dataset=poetryT5.dataset:main',
        'pt5-preprocess=poetryT5.preprocess:main',
        'pt5-train=poetryT5.train:main'
      ]
  }
)