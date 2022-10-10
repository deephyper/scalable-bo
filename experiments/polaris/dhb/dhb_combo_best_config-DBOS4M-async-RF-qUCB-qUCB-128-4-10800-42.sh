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

export OUTPUT_DIR="dhb_combo-DBOS4M-async-RF-qUCB-qUCB-128-4-10800-42"
export STDOUT_FILE="output/$OUTPUT_DIR/training_stdout.txt"
export STDERR_FILE="output/$OUTPUT_DIR/training_stderr.txt"

python dhb_combo_best_config-DBOS4M-async-RF-qUCB-qUCB-128-4-10800-42.py >  $STDOUT_FILE 2> $STDERR_FILE

