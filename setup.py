from setuptools import setup, find_packages
import os

with open('README.md') as f:
    readme = f.read()

setup(
    name='cyclicmodel',
    description='statistical causality discovery based on cyclic model',
    version='0.0.4',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Akimitsu INOUE',
    author_email='akimitsu.inoue@gmail.com',
    url='https://github.com/inoueakimitsu/cyclicmodel',
    packages=find_packages(),
    install_requires=['theano', 'numpy', 'pymc3'],
    test_suite='tests',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],

)
