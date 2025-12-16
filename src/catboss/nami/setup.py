from setuptools import setup, find_packages, Extension
import sys
import os
import pybind11

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# C++ extension module - MANDATORY for NAMI performance
cpp_sources = [
    'nami/cpp/nami_core.cpp',
    'nami/cpp/uv_calc.cpp',
    'nami/cpp/data_collection.cpp',
    'nami/cpp/polynomial_fit.cpp',
    'nami/cpp/outlier_detection.cpp',
    'nami/cpp/spline_fit.cpp',
]

# Compiler flags for maximum performance
extra_compile_args = ['-O3', '-std=c++11']
extra_link_args = []

# Platform-specific optimizations
if sys.platform == 'darwin':  # macOS
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.14']
elif sys.platform == 'linux':
    extra_compile_args += ['-fopenmp']
    extra_link_args += ['-fopenmp']

# SIMD optimizations for maximum speed
if sys.platform in ['linux', 'darwin']:
    extra_compile_args += ['-march=native']

ext_modules = [
    Extension(
        '_nami_core',
        cpp_sources,
        include_dirs=[
            pybind11.get_include(),
            'nami/cpp',
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]


setup(
    name="nami",
    version="0.2.0",
    author="Arpan Pal",
    author_email="apal@ncra.tifr.res.in",
    description="NAMI: Nonlinear Automated Monotonous filter for Interference (C++ accelerated)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/arpan-52/nami",
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=requirements,
    setup_requires=['pybind11>=2.6.0'],
    entry_points={
        "console_scripts": [
            "nami = nami.navigator:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires=">=3.7",
    zip_safe=False,
)
