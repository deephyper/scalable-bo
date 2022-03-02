#!/bin/bash

export MAX_EVALS=100
export RANDOM_STATES=(42 2022 1451 8317 213 7607 4978 1516 2335 3366)
export PROBLEMS=("ackley") 

# AMBS + Liar Strategy
for random_state in ${RANDOM_STATES[@]}; do
    for problem in ${PROBLEMS[@]}; do
        echo "Running: python exp.py --problem $problem --search AMBS --sync 0 --timeout 5 --verbose 1 --random-state $random_state --strategy boltzmann";
        python exp.py --problem $problem --search AMBS --sync 0 --timeout 5 --verbose 1 --random-state $random_state --strategy boltzmann
        done
    done
done