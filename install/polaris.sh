#!/bin/bash
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:20:00
#PBS -q debug 
#PBS -A datascience

cd ${PBS_O_WORKDIR}

module load conda/2022-07-19

conda create -p dhenv --clone base -y
conda activate dhenv/

# Clone DeepHyper (develop)
git clone -b develop git@github.com:deephyper/deephyper.git

# Install DeepHyper
pip install -e deephyper/

# Install Scalable-BO
pip install -e ../src/scalbo/

# Copy activation of environment file
cp ../install/env/polaris.sh activate-dhenv.sh
echo "" >> activate-dhenv.sh
echo "conda activate $PWD/dhenv/" >> activate-dhenv.sh
