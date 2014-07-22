#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

# Hackishly inject a constant into builtins to enable importing of the
# package before the library is built.
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__GEORGE_SETUP__ = True
import george_ess

setup(
    name="george_ess",
    version=george_ess.__version__,
    author="Daniel Foreman-Mackey",
    author_email="danfm@nyu.edu",
    url="https://github.com/dfm/george-ess",
    license="MIT",
    py_modules=["george_ess"],
    description="Elliptical slice sampling for George",
    long_description=open("README.rst").read(),
    package_data={"": ["README.rst", "LICENSE"]},
    include_package_data=True,
    classifiers=[
        # "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
