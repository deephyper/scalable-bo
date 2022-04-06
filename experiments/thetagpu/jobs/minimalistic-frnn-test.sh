#!/bin/bash
#COBALT -n 1
#COBALT -t 20
#COBALT -q single-gpu
#COBALT -A datascience

source ../../../build/activate-dhenv.sh

export RANKS_PER_NODE=1
export COBALT_JOBSIZE=1
export PYTHONPATH=../../../build/dhenv/lib/python3.8/site-packages/:$PYTHONPATH


echo "Running: mpirun -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH -n $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) --hostfile $COBALT_NODEFILE python -m scalbo.benchmark.minimalistic_frnn"
mpirun -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH -np $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) --hostfile $COBALT_NODEFILE python -m scalbo.benchmark.minimalistic_frnn