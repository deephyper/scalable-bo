#!/bin/bash

set -xe

export problem="dhb_proteinstructure"
export max_evals=200
export pruning_strategies=("CONST1" "CONST4")
# export pruning_strategies=("NONE" "SHA" "MED" "LOGLIN2" "ULOGLIN2") 
# export pruning_strategies=("CONST1" "CONST2" "CONST4" "CONST8")
export random_states=(1608637542 3421126067 4083286876  787846414 3143890026 3348747335 2571218620 2563451924  670094950 1914837113) 

exec_search () {
    export log_dir="output/$problem-RANDOM-B$pruning_strategy-$max_evals-$random_state"
    mkdir -p $log_dir

    mpirun -np 1 python -m scalbo.exp --problem $problem \
        --search DBO \
        --model DUMMY \
        --max-evals $max_evals \
        --random-state $random_state \
        --log-dir $log_dir \
        --pruning-strategy $pruning_strategy \
        --max-steps 100 \
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
