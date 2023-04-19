#!/bin/bash

#!!! CONFIGURATION - START
export problem="dhb_navalpropulsion"
export search="DBO"
export model="RF"
export acq_func="UCB"
export scheduler_periode=48
export scheduler_rate=0.1
export pruning_strategy="SHA"
export objective_scaler="minmaxlog"
export timeout=120
export random_state=42
#!!! CONFIGURATION - END

export log_dir="output/$problem-$search-$model-$acq_func-$pruning_strategy-$NUM_WORKERS-$timeout-$random_state"
mkdir -p $log_dir

# Setup Redis Database
pushd $log_dir
redis-server ~/.redis.conf &
export DEEPHYPER_DB_HOST=localhost
popd

sleep 1

mpirun -np $NUM_WORKERS  --host localhost:$NUM_WORKERS \
    python -m scalbo.exp --problem $problem \
    --search $search \
    --model $model \
    --acq-func $acq_func \
    --objective-scaler $objective_scaler \
    --scheduler 1 \
    --scheduler-periode $scheduler_periode \
    --scheduler-rate $scheduler_rate \
    --random-state $random_state \
    --log-dir $log_dir \
    --pruning-strategy $pruning_strategy \
    --timeout $timeout \
    --max-steps 100 \
    --interval-steps 1 \
    --filter-duplicated 0 \
    --n-jobs 1

redis-cli shutdown