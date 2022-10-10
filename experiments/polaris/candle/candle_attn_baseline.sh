#!/bin/bash
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:60:00
#PBS -q debug 
#PBS -A datascience

cd ${PBS_O_WORKDIR}

source ../../../build/activate-dhenv.sh

python -m scalbo.benchmark.candle_attn > candle_attn_baseline_stdout.txt 2> candle_attn_baseline_stderr.txt
