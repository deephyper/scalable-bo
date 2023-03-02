#!/bin/bash

set -xe

export problem="dhb_ipl"
export search="DBO"
export model="RF"
export acq_func="UCB"
export scheduler_periode=48
export scheduler_rate=0.1
export max_evals=200
export pruning_strategy="SHA"
export objective_scaler="minmaxlog"
# export random_states=(42)
export random_states=(1608637542 3421126067 4083286876  787846414 3143890026 3348747335 2571218620 2563451924  670094950 1914837113) 

exec_search () {
    export log_dir="output/$problem-$search-$pruning_strategy-$model-alc-$acq_func-$objective_scaler-scheduler-$scheduler_periode-$scheduler_rate-delay-$max_evals-$random_state"
    mkdir -p $log_dir

    mpirun -np 1 python -m scalbo.exp --problem $problem \
        --search $search \
        --model $model \
        --acq-func $acq_func \
        --objective-scaler $objective_scaler \
        --scheduler 1 \
        --scheduler-periode $scheduler_periode \
        --scheduler-rate $scheduler_rate \
        --max-evals $max_evals \
        --random-state $random_state \
        --log-dir $log_dir \
        --pruning-strategy $pruning_strategy \
        --max-steps 1000 \
        --interval-steps 10
}

for random_state in ${random_states[@]}; do
   exec_search
done


