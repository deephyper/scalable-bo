#!/bin/bash

#!!! CONFIGURATION - START
export problem="dhb_navalpropulsion"
export search="OPT-NSGAII"
export pruning_strategy="NONE"
export timeout=5
export random_state=42
#!!! CONFIGURATION - END

export NRANKS_PER_NODE=4
export NDEPTH=$((8  / $NRANKS_PER_NODE))
export NNODES=1
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE ))
export OMP_NUM_THREADS=$NDEPTH
export OPTUNA_N_OBJECTIVES=2

export DEEPHYPER_LOG_DIR="output/$problem-$search-$pruning_strategy-$NNODES-$timeout-$random_state"
mkdir -p $DEEPHYPER_LOG_DIR

### Setup Postgresql Database - START ###
export OPTUNA_DB_DIR="$DEEPHYPER_LOG_DIR/optunadb"
export OPTUNA_DB_HOST="localhost"
initdb -D "$OPTUNA_DB_DIR"
pg_ctl -D $OPTUNA_DB_DIR -l "$DEEPHYPER_LOG_DIR/db.log" start
createdb hpo
### Setup Postgresql Database - END ###

sleep 5

mpiexec -np ${NTOTRANKS} \
    python -m scalbo.exp \
    --problem $problem \
    --search $search \
    --random-state $random_state \
    --log-dir $DEEPHYPER_LOG_DIR \
    --pruning-strategy $pruning_strategy \
    --timeout $timeout \
    --max-steps 100 

# Drop and stop the database
dropdb hpo
pg_ctl -D $OPTUNA_DB_DIR -l "$DEEPHYPER_LOG_DIR/db.log" stop