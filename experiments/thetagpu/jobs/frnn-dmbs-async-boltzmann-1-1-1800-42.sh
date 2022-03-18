#!/bin/bash
#COBALT -n 1
#COBALT -t 20
#COBALT -q full-node
#COBALT -A datascience

source ../../../build/activate-dhenv.sh

export RANKS_PER_NODE=4
export COBALT_JOBSIZE=1
export PYTHONPATH=../../../build/dhenv/lib/python3.8/site-packages/:$PYTHONPATH


export liar_strategy="boltzmann"
export timeout=1800
export random_state=42 
export problem="frnn"
export cache_dir="/dev/shm"
export sync_val=0
export search="DMBS"

if [[ "$sync_val" -eq 0 ]];
then
  export sync_str="async"
else
  export sync_str="sync"
fi

# AMBS
export log_dir="output/$problem-$search-$sync_str-$liar_strategy-$COBALT_JOBSIZE-$RANKS_PER_NODE-$timeout-$random_state";

echo "mpirun -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH -np $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) -npernode $RANKS_PER_NODE --hostfile $COBALT_NODEFILE python -m scalbo.exp --problem $problem \
--search $search \
--sync $sync_val \
--timeout $timeout \
--strategy $liar_strategy \
--random-state $random_state \
--log-dir $log_dir \
--verbose 1";

mpirun -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH -np $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) -npernode $RANKS_PER_NODE --hostfile $COBALT_NODEFILE python -m scalbo.exp --problem $problem \
    --search $search \
    --sync $sync_val \
    --timeout $timeout \
    --strategy $liar_strategy \
    --random-state $random_state \
    --log-dir $log_dir \
    --verbose 1 