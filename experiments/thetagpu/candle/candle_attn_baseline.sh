#!/bin/bash
#COBALT -n 1
#COBALT -t 60
#COBALT -q single-gpu
#COBALT -A datascience


source ../../../build/activate-dhenv.sh

python -m scalbo.benchmark.candle_attn > candle_baseline_stdout.txt 2> candle_baseline_stderr.txt
