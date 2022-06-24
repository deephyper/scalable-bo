#!/bin/bash
#COBALT -n 1
#COBALT -t 60
#COBALT -q single-gpu
#COBALT -A datascience


source ../../../build/activate-dhenv.sh
export best_config="output/candle_attn-HB-async-DUMMY-UCB-cl_max-8-8-25200-42/best-config.json"

python -m scalbo.benchmark.candle_attn --json $best_config > candle_best_HB_stdout.txt 2> candle_best_HB_stderr.txt
