
   
from setuptools import setup

REQUIRED = [
    "matplotlib",
    "pyaml",
    "optuna"
]


setup(name="scalbo", packages=["scalbo"], install_requires=REQUIRED)