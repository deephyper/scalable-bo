#!/bin/bash

set -x

export NRANKS_PER_NODE=8
export acq_func="UCB"
export strategy="cl_max"
export model="GP"
export timeout=20
export random_state=42 
export problem="test"
export sync_val=0
export search="CBO"

if [[ "$sync_val" -eq 0 ]];
then
  export sync_str="async"
else
  export sync_str="sync"
fi

export log_dir="output/$problem-$search-$sync_str-$model-$acq_func-$strategy-1-$NRANKS_PER_NODE-$timeout-$random_state"
mkdir $log_dir



mpirun -np $NRANKS_PER_NODE python -m scalbo.exp --problem $problem \
    --search $search \
    --model $model \
    --sync $sync_val \
    --timeout $timeout \
    --acq-func $acq_func \
    --strategy $strategy \
    --random-state $random_state \
    --log-dir $log_dir \
    --verbose 1
    
