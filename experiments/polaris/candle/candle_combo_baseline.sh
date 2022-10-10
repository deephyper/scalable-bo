#!/bin/bash
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:60:00
#PBS -q debug
#PBS -A datascience
#PBS -l filesystems=grand:home

cd ${PBS_O_WORKDIR}

source ../../../build/activate-dhenv.sh

python -m scalbo.benchmark.candle_combo > candle_combo_baseline_stdout.txt 2> candle_combo_baseline_stderr.txt
