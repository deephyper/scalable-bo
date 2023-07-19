#!/bin/bash

#!!! CONFIGURATION - START
export problem="dhb_navalpropulsion"
export search="OPT-TPE"
export max_evals=200
#!!! CONFIGURATION - END

export random_states=(1608637542 3421126067 4083286876  787846414 3143890026 3348747335 2571218620 2563451924  670094950 1914837113)

export NRANKS=1
export OPTUNA_N_OBJECTIVES=2
export DEEPHYPER_BENCHMARK_MOO=1

### Setup Postgresql Database - START ###
export OPTUNA_DB_DIR="optunadb"
export OPTUNA_DB_HOST="localhost"
initdb -D "$OPTUNA_DB_DIR"
pg_ctl -D $OPTUNA_DB_DIR -l "db.log" start
### Setup Postgresql Database - END ###

sleep 5

exec_search () {
    createdb hpo

    export DEEPHYPER_LOG_DIR="output-tpe/$problem-$search-$max_evals-$random_state"
    mkdir -p $DEEPHYPER_LOG_DIR

    mpiexec -np ${NRANKS} \
        python -m scalbo.exp \
        --problem $problem \
        --search $search \
        --random-state $random_state \
        --log-dir $DEEPHYPER_LOG_DIR \
        --max-evals $max_evals\
        --max-steps 100 

    dropdb hpo
}

for random_state in ${random_states[@]}; do
    exec_search;
    sleep 5;
done

# Drop and stop the database
pg_ctl -D $OPTUNA_DB_DIR -l "db.log" stop