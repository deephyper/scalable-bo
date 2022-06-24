#!/bin/bash
#COBALT -n 8
#COBALT -t 425
#COBALT -q full-node
#COBALT -A datascience

#!!! CONFIGURATION - START
export RANKS_PER_NODE=8
export COBALT_JOBSIZE=8
export acq_func="UCB"
export strategy="cl_max"
export model="GP"
export timeout=25200
export random_state=42 
export problem="candle_attn"
export sync_val=0
export search="CBO"
#!!! CONFIGURATION - END

# activate Python environment
source ../../../build/activate-dhenv.sh
export PYTHONPATH=../../../build/dhenv/lib/python3.8/site-packages/:$PYTHONPATH

# # start MPS daemon on each node
# ./launch-mps-service.sh

# # For MPS Client Application
# export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps 
# export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

if [[ "$sync_val" -eq 0 ]];
then
  export sync_str="async"
else
  export sync_str="sync"
fi

# AMBS
export log_dir="output/$problem-$search-$sync_str-$model-$acq_func-$strategy-$COBALT_JOBSIZE-$RANKS_PER_NODE-$timeout-$random_state";

# profile gpu
mkdir $log_dir;
gpustat --no-color -i >> "$log_dir/gpustat.txt" &

echo "mpirun -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH -np $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) --hostfile $COBALT_NODEFILE python -m scalbo.exp --problem $problem \
--search $search \
--model $model \
--sync $sync_val \
--timeout $timeout \
--acq-func $acq_func \
--strategy $strategy \
--random-state $random_state \
--log-dir $log_dir \
--verbose 1";

mpirun -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH -n $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) -N $RANKS_PER_NODE --hostfile $COBALT_NODEFILE python -m scalbo.exp --problem $problem \
    --search $search \
    --model $model \
    --sync $sync_val \
    --timeout $timeout \
    --acq-func $acq_func \
    --strategy $strategy \
    --random-state $random_state \
    --log-dir $log_dir \
    --verbose 1 
