#!/bin/bash
#COBALT -n 128
#COBALT -t 40
#COBALT -q default
#COBALT -A datascience

PROJECT=~/projects/grand/deephyper/search_quality
INIT_SCRIPT=$PROJECT/scripts/init_dh-mpi.sh

source $INIT_SCRIPT

export RANKS_PER_NODE=8

export LIAR_STRATEGIES=("cl_max" "boltzmann" "topk")

export timeout=300
export RANDOM_STATES=(42)
export PROBLEMS=("ackley")
declare -A COMMUNICATION
COMMUNICATION["sync"]=1
COMMUNICATION["async"]=0
export cache_dir="/dev/shm"

# AMBS
for comm in ${!COMMUNICATION[@]}; do
    for liar_strategy in ${LIAR_STRATEGIES[@]}; do
        for random_state in ${RANDOM_STATES[@]}; do
            for problem in ${PROBLEMS[@]}; do
                export log_dir="output/$problem-ambs-$comm-$liar_strategy-$COBALT_JOBSIZE-$RANKS_PER_NODE-$timeout-$random_state";
                echo "Running: aprun -r 1 -n $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) -N $RANKS_PER_NODE -j 4 python -m scalbo.exp --problem $problem \
                --search AMBS \
                --sync ${COMMUNICATION[$comm]} \
                --timeout $timeout \
                --liar-strategy $liar_strategy \
                --random-state $random_state \
                --log-dir $log_dir \
                --cache-dir $cache_dir \
                --verbose 1";
                aprun -r 1 -n $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) -N $RANKS_PER_NODE -j 4 python -m scalbo.exp --problem $problem \
                    --search AMBS \
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