#!/bin/bash
#COBALT -n 2
#COBALT -t 30
#COBALT -q debug-flat-quad
#COBALT -A datascience

PROJECT=~/projects/grand/deephyper/search_quality
INIT_SCRIPT=$PROJECT/scripts/init_dh-mpi.sh

source $INIT_SCRIPT

export RANKS_PER_NODE=4

export LIAR_STRATEGIES=("boltzmann")

export timeout=50
export RANDOM_STATES=(42) # 2022 1451 8317 213 7607 4978 1516 2335 3366)
export PROBLEMS=("ackley")
declare -A COMMUNICATION
# COMMUNICATION["sync"]=1
COMMUNICATION["async"]=0
export cache_dir="/dev/shm"

# AMBS
for comm in ${!COMMUNICATION[@]}; do
    for liar_strategy in ${LIAR_STRATEGIES[@]}; do
        for random_state in ${RANDOM_STATES[@]}; do
            for problem in ${PROBLEMS[@]}; do
                export log_dir="output/$problem-dmbs-$comm-$liar_strategy-$COBALT_JOBSIZE-$RANKS_PER_NODE-$timeout-$random_state";
                echo "Running: aprun -n $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) -N $RANKS_PER_NODE -d 8 -j 4 -cc depth -e OMP_NUM_THREADS=8 python -m scalbo.exp --problem $problem \
                --search DMBS \
                --sync ${COMMUNICATION[$comm]} \
                --timeout $timeout \
                --liar-strategy $liar_strategy \
                --random-state $random_state \
                --log-dir $log_dir \
                --cache-dir $cache_dir \
                --verbose 1";
                aprun -n $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) -N $RANKS_PER_NODE -d 8 -j 4 -cc depth -e OMP_NUM_THREADS=8 python -m scalbo.exp --problem $problem \
                    --search DMBS \
                    --sync ${COMMUNICATION[$comm]} \
                    --timeout $timeout \
                    --liar-strategy $liar_strategy \
                    --random-state $random_state \
                    --log-dir $log_dir \
                    --cache-dir $cache_dir \
                    --verbose 1 
            done
        done
    done
done