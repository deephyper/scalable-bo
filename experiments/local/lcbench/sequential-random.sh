#!/bin/bash

set -xe

export problem="dhb_lcbench"
# export dataset="APSFailure"
export max_evals=200
export pruning_strategies=("NONE" "CONST1" "SHA")
export random_states=(1608637542 3421126067 4083286876  787846414 3143890026 3348747335 2571218620 2563451924  670094950 1914837113) 

exec_search () {
    export log_dir="output/$dataset-RANDOM-$pruning_strategy-$max_evals-$random_state"
    mkdir -p $log_dir

    export DEEPHYPER_BENCHMARK_LCBENCH_DATASET=$dataset

    python -m scalbo.exp --problem $problem \
        --search BO \
        --model DUMMY \
        --max-evals $max_evals \
        --random-state $random_state \
        --log-dir $log_dir \
        --pruning-strategy $pruning_strategy \
        --max-steps 50 \
        --interval-steps 1 
}

for pruning_strategy in ${pruning_strategies[@]}; do
  for random_state in ${random_states[@]}; do
    for i in {1..5}; do 
        exec_search && break || sleep 5;
    done
    sleep 1;
  done
done


