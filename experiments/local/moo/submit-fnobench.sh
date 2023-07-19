#!/bin/bash

#!!! CONFIGURATION - START
export problem="dhb_fnobench"
export max_evals=200
#!!! CONFIGURATION - END

export random_states=(42)

export NRANKS=1
export DEEPHYPER_BENCHMARK_MOO=1
export DEEPHYPER_DB_HOST=localhost

sleep 5

exec_search () {

    export DEEPHYPER_LOG_DIR="output-fnobench/$problem-DBO-$max_evals-$random_state"
    mkdir -p $DEEPHYPER_LOG_DIR

    mpiexec -np ${NRANKS} \
        python -m scalbo.exp \
        --problem $problem \
        --search DBO \
        --model RF \
        --acq-func UCB \
        --scheduler 1 \
        --scheduler-periode 48 \
        --scheduler-rate 0.1 \
        --filter-duplicated 1 \
        --max-evals $max_evals \
        --random-state $random_state \
        --log-dir $DEEPHYPER_LOG_DIR \
        --max-steps 100  

}

for random_state in ${random_states[@]}; do
    exec_search;
    sleep 5;
done
