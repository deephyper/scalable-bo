#!/bin/bash

#!!! CONFIGURATION - START
export problem="dhb_navalpropulsion"
export max_evals=200
#!!! CONFIGURATION - END

export random_states=(1608637542 3421126067 4083286876  787846414 3143890026 3348747335 2571218620 2563451924  670094950 1914837113)

export NRANKS=1
export DEEPHYPER_DB_HOST=localhost
export DEEPHYPER_BENCHMARK_MOO=1

sleep 5

exec_search () {

    export DEEPHYPER_LOG_DIR="output-random/$problem-RANDOM-$max_evals-$random_state"
    mkdir -p $DEEPHYPER_LOG_DIR

    mpiexec -np ${NRANKS} \
        python -m scalbo.exp \
        --problem $problem \
        --search DBO \
        --model DUMMY \
        --acq-func UCB \
        --max-evals $max_evals \
        --random-state $random_state \
        --log-dir $DEEPHYPER_LOG_DIR \
        --max-steps 100  

}

for random_state in ${random_states[@]}; do
    exec_search;
    sleep 5;
done
