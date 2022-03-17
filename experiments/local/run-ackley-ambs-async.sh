#!/bin/bash

# export RANDOM_STATES=(42 2022 1451 8317 213 7607 4978 1516 2335 3366)
export STRATEGIES=("cl_max")
export timeout=30
export RANDOM_STATES=(42)
export PROBLEMS=("ackley") 
export acq_func="UCB"

# AMBS + Liar Strategy
for strategy in ${STRATEGIES[@]}; do
    for random_state in ${RANDOM_STATES[@]}; do
        for problem in ${PROBLEMS[@]}; do
            export log_dir="output/$problem-ambs-async-$acq_func-$strategy-1-8-$timeout-$random_state";
            echo "Running: mpirun -np 8 python -m scalbo.exp --problem $problem \
            --search AMBS \
            --timeout $timeout \
            --acq-func $acq_func \
            --strategy $strategy \
            --random-state $random_state \
            --log-dir $log_dir \
            --verbose 1";
            mpirun -np 8 python -m scalbo.exp --problem $problem \
                --search AMBS \
                --timeout $timeout \
                --acq-func $acq_func \
                --strategy $strategy \
                --random-state $random_state \
                --log-dir $log_dir \
                --verbose 1 
        done
    done
done