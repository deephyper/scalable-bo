#!/bin/bash

# Download Spack and Install
git clone -c feature.manyFiles=true https://github.com/spack/spack.git spack
. spack/share/spack/setup-env.sh

# Download Mochi Packages Repo 
git clone https://github.com/mochi-hpc/mochi-spack-packages.git

# Create dh-env 
spack env create dhenv spack-dhenv.yaml
spack env activate dhenv
spack install