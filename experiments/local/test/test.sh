#!/bin/bash

set -xe

export NRANKS_PER_NODE=4
export random_state=42 
export problem="test"

if [[ "$sync_val" -eq 0 ]];
then
  export sync_str="async"
else
  export sync_str="sync"
fi

export log_dir="output/$problem-$search-$sync_str-$model-$acq_func-$strategy-1-$NRANKS_PER_NODE-$timeout-$random_state"
mkdir -p $log_dir

# Create database
export DEEPHYPER_DB_HOST="localhost"
redis-server $(spack find --path redisjson | grep -o "/.*/redisjson.*")/redis.conf &

sleep 5

mpirun -np $NRANKS_PER_NODE python -m scalbo.exp --problem $problem \
    --search DBO \
    --model RF \
    --acq-func UCB \
    --objective-scaler minmaxlog \
    --scheduler 1 \
    --scheduler-periode 48 \
    --scheduler-rate 0.1 \
    --max-evals 100 \
    --random-state $random_state \
    --log-dir $log_dir \
    --pruning-strategy SHA \
    --max-steps 50 \
    --interval-steps 1 \
    --filter-duplicated 1

redis-cli shutdown
