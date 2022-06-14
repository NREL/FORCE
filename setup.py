""""Distribution setup"""

import os

from setuptools import setup, find_packages

import versioneer

ROOT = os.path.abspath(os.path.dirname(__file__))

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name="FORCE",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
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
