#!/bin/bash
#COBALT -n 256
#COBALT -t 45
#COBALT -q default
#COBALT -A datascience

source ../../../build/activate-dhenv.sh

export RANKS_PER_NODE=32
export liar_strategy="boltzmann"
export timeout=1800
export random_state=42 
export problem="ackley"
export cache_dir="/dev/shm"
export sync_val=1
export search="DMBS"

if [[ "$sync_val" -eq 0 ]];
then
  export sync_str="async"
else
  export sync_str="sync"
fi

# AMBS
export log_dir="output/$problem-$search-$sync_str-$liar_strategy-$COBALT_JOBSIZE-$RANKS_PER_NODE-$timeout-$random_state";

echo "Running: aprun -n $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) -N $RANKS_PER_NODE -d 8 -j 4 -cc depth -e OMP_NUM_THREADS=8 python -m scalbo.exp --problem $problem \
--search $search \
--sync $sync_val \
--timeout $timeout \
--liar-strategy $liar_strategy \
--random-state $random_state \
--log-dir $log_dir \
--verbose 1";

aprun -n $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) -N $RANKS_PER_NODE -d 8 -j 4 -cc depth -e OMP_NUM_THREADS=8 python -m scalbo.exp --problem $problem \
    --search $search \
    --sync $sync_val \
    --timeout $timeout \
    --liar-strategy $liar_strategy \
    --random-state $random_state \
    --log-dir $log_dir \
    --verbose 1 
