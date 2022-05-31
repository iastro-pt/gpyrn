#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup


# Get the long description from the README file
from os import path
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gpyrn',
    version='1.0.1',
    description='Gaussian process regression networks for exoplanet detection',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='João Camacho',
    author_email='joao.camacho@astro.up.pt',
    license='MIT',
    url='https://github.com/iastro-pt/gpyrn',
    packages=['gpyrn'],
)
