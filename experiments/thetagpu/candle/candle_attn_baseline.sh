#!/bin/bash
#COBALT -n 1
#COBALT -t 60
#COBALT -q single-gpu
#COBALT -A datascience


source ../../../build/activate-dhenv.sh

python -m scalbo.benchmark.candle_att > candle_attn_baseline.txt