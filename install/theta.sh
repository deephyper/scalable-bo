#!/bin/bash

module load miniconda-3
module load gcc/8.3.0
module load cray-mpich

conda create -p dhenv python=3.8 -y
conda activate dhenv/

# Clone DeepHyper (develop)
git clone -b develop https://github.com/deephyper/deephyper.git

# Clone DeepHyper/Scikit-Optimize (master)
git clone https://github.com/deephyper/scikit-optimize.git

# Install DeepHyper
pip install ConfigSpace
pip install -e scikit-optimize/
pip install -e deephyper/

# Install MPI4PY
git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py/
sed -i "s/## mpicc .*/mpicc = cc/g" mpi.cfg
sed -i "s/## mpicxx .*/mpicxx = CC/g" mpi.cfg
CC=cc CXX=CC python setup.py build
CC=cc CXX=CC python setup.py install
cd ..

# Install Scalable-BO
pip install -e ../src/scalbo/

# Install rdkit
pip install rdkit-pypi

# Copy activation of environment file
cp ../install/env/theta.sh activate-dhenv.sh
echo "conda activate $PWD/dhenv/" >> activate-dhenv.sh
