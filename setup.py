#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup

setup(name='gpyrn',
      version='1.0',
      description='Modelling stellar activity in radial velocity data with Gaussian processes regression networks ',
      author='João Camacho',
      author_email='joao.camacho@astro.up.pt',
      license='MIT',
      url='https://github.com/jdavidrcamacho/gpyrn',
      packages=['gpyrn'],
      install_requires=[
        'numpy',
        'scipy',
      ],
     )
