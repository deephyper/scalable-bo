
   
from setuptools import setup

REQUIRED = [
    "matplotlib",
    "pyaml"
]


setup(name="scalbo", packages=["scalbo"], install_requires=REQUIRED)