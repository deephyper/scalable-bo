#!/bin/bash

# . /etc/profile

set -e

module load cmake
module load open-ce/1.5.0-py38-0
#module load open-ce

conda create -p dhenv --clone open-ce-1.5.0-py38-0 -y
#conda create -p dhenv --clone open-ce-1.5.2-py39-0 -y
conda activate dhenv/

# Clone DeepHyper (develop)
git clone -b develop https://github.com/deephyper/deephyper.git

# Clone DeepHyper/Scikit-Optimize (master)
git clone https://github.com/deephyper/scikit-optimize.git

# Clone Plasma Fork (tf2)
git clone -b tf2 https://github.com/deephyper/plasma-python.git

# Install DeepHyper
pip install -e deephyper/
pip install -e scikit-optimize/
pip install -e plasma-python/

# Install Scalable-BO
pip install -e ../src/scalbo/

# Install mpi4py
git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py/

module load gcc

MPICC=mpicc python setup.py install
cd ..

# Install rdkit
####pip install rdkit-pypi

# Copy activation of environment file
cp ../install/env/summit.sh activate-dhenv.sh
echo "conda activate $PWD/dhenv/" >> activate-dhenv.sh
