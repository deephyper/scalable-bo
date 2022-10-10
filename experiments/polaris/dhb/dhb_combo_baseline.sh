#!/bin/bash
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -A datascience
#PBS -l filesystems=grand:home

set -x

cd ${PBS_O_WORKDIR}

source ../../../build/activate-dhenv.sh

mkdir -p dh_combo_baseline/
cd dh_combo_baseline/

python ../dhb_combo_baseline.py > dh_combo_baseline_stdout.txt 2> dh_combo_baseline_stderr.txt
