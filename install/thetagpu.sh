#!/bin/bash

. /etc/profile

module load conda/2021-11-30

conda create -p dhenv --clone base -y
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
env MPICC=/lus/theta-fs0/software/thetagpu/openmpi/openmpi-4.1.1_ucx-1.11.2_gcc-9.3.0/bin/mpicc pip install -I mpi4py --no-cache-dir

# Copy activation of environment file
cp ../install/env/thetagpu.sh activate-dhenv.sh
echo "conda activate $PWD/dhenv/" >> activate-dhenv.sh
