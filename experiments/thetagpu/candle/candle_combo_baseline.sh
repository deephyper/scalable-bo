#!/bin/bash
#COBALT -n 1
#COBALT -t 60
#COBALT -q single-gpu
#COBALT -A datascience
#COBALT --attrs filesystems:home,grand,theta-fs0


source ../../../build/activate-dhenv.sh

python -m scalbo.benchmark.candle_combo > candle_combo_baseline_stdout.txt 2> candle_combo_baseline_stderr.txt
