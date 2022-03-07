#!/bin/bash
#COBALT -n 2
#COBALT -t 15
#COBALT -q debug-cache-quad
#COBALT -A datascience

PROJECT=~/projects/grand/deephyper/search_quality
INIT_SCRIPT=$PROJECT/scripts/init_dh-mpi.sh

source $INIT_SCRIPT

export RANKS_PER_NODE=2

aprun -n $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) -N $RANKS_PER_NODE -d 8 -j 4 -cc depth -e OMP_NUM_THREADS=8 python test-aprun.py --problem ackley \
    --search AMBS \
    --sync 0 \
    --timeout 10 \
    --liar-strategy boltzmann \
    --random-state 42 \
    --log-dir output \
    --cache-dir /dev/shm \
    --verbose 1 