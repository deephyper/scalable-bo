#!/bin/bash

set -xe

export problem="dhb_ipl"
export max_evals=200
# export pruning_strategies=("MMF4" "UMMF4")
export pruning_strategies=("UMMF4")
# export random_states=(42)
export random_states=(1608637542 3421126067 4083286876  787846414 3143890026 3348747335 2571218620 2563451924  670094950 1914837113) 

exec_search () {
    export log_dir="output/$problem-RANDOM-$pruning_strategy-$max_evals-$random_state"
    mkdir -p $log_dir

    mpirun -np 1 python -m scalbo.exp --problem $problem \
        --search DBO \
        --model DUMMY \
        --max-evals $max_evals \
        --random-state $random_state \
        --log-dir $log_dir \
        --pruning-strategy $pruning_strategy \
        --max-steps 1000 \
        --interval-steps 10
}

for pruning_strategy in ${pruning_strategies[@]}; do
    for random_state in ${random_states[@]}; do
        exec_search
    done
done


