""""Distribution setup"""

import os
import codecs

from setuptools import setup, find_packages

ROOT = os.path.abspath(os.path.dirname(__file__))

with open("README.rst", "r") as fh:
    long_description = fh.read()

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name="FORCE",
    version=get_version("package/__init__.py"),
    description="Forecasting Offshore wind Reductions in Cost of Energy",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "pandas",
        "pyyaml",
        "xlsxwriter",
        "orbit-nrel"
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "pylint",
            "flake8",
            "black",
            "isort",
            "pytest",
            "pytest-cov",
            "sphinx",
            "sphinx-rtd-theme",
        ]
    },
    test_suite="pytest",
    tests_require=["pytest", "pytest-cov"],
)
