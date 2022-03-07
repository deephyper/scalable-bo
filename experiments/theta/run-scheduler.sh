#!/bin/bash
#COBALT -n 2
#COBALT -t 20
#COBALT -q debug-cache-quad
#COBALT -A datascience

PROJECT=~/projects/grand/deephyper/search_quality
INIT_SCRIPT=$PROJECT/scripts/init_dh-mpi.sh

source $INIT_SCRIPT

export RANKS_PER_NODE=2
export configs="configs.yaml"

echo "Running: aprun -n $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) -N $RANKS_PER_NODE -d 8 -j 4 -cc depth -e OMP_NUM_THREADS=8 python scheduler.py --configs $configs \
    --num-nodes $COBALT_JOBSIZE \
    --ranks-per-node $RANKS_PER_NODE"
aprun -n $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) -N $RANKS_PER_NODE -d 8 -j 4 -cc depth -e OMP_NUM_THREADS=8 python scheduler.py --configs $configs \
    --num-nodes $COBALT_JOBSIZE \
    --ranks-per-node $RANKS_PER_NODE