from setuptools import setup, find_packages
import os

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='cyclicmodel',
    description='statistical causality discovery based on cyclic model',
    version='0.0.1',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Akimitsu INOUE',
    author_email='akimitsu.inoue@gmail.com',
    url='https://github.com/inoueakimitsu/cyclicmodel',
    license=license,
    packages=find_packages(),
    install_requires=['theano', 'numpy', 'pymc3'],
    test_suite='tests',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],

)
