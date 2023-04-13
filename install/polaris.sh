#!/bin/bash

set -xe

module load PrgEnv-gnu/8.3.3
module load llvm/release-15.0.0
module load conda/2022-09-08

conda create -p dhenv --clone base -y
conda activate dhenv/
pip install --upgrade pip

# Install Spack
git clone -c feature.manyFiles=true https://github.com/spack/spack.git
. ./spack/share/spack/setup-env.sh

git clone git@github.com:deephyper/deephyper-spack-packages.git

# Install RedisJson With Spack
spack env create redisjson
spack env activate redisjson
spack repo add deephyper-spack-packages
spack add redisjson
spack install

# Clone DeepHyper (develop)
git clone -b develop git@github.com:deephyper/deephyper.git

# Install DeepHyper with MPI and Redis backends
pip install -e "deephyper/[default,mpi,redis]"

# Install Benchmarks
git clone git@github.com:deephyper/benchmark.git deephyper-benchmark
pip install -e "deephyper-benchmark/"

# Install Scalable-BO
pip install -e ../src/scalbo/

pip install gpustats

# Copy activation of environment file
cp ../install/env/polaris.sh activate-dhenv.sh
echo "" >> activate-dhenv.sh
echo "conda activate $PWD/dhenv/" >> activate-dhenv.sh

# Activate Spack env
echo "" >> activate-dhenv.sh
echo ". $PWD/spack/share/spack/setup-env.sh" >> activate-dhenv.sh
echo "spack env activate redisjson" >> activate-dhenv.sh

# Redis Configuration
cp ../install/env/redis.conf redis.confg
cat $(spack find --path redisjson | grep -o "/.*/redisjson.*")/redis.conf >> redis.conf

# Install the Combo Benchmark
python -c "from deephyper_benchmark import *; install('ECP-Candle/Pilot1/Combo');"