#!/bin/bash
#COBALT -n 1
#COBALT -t 190
#COBALT -q bigmem
#COBALT -A datascience

#!!! CONFIGURATION - START
export RANKS_PER_NODE=48
export COBALT_JOBSIZE=1
export acq_func="qUCB"
export strategy="qUCB"
export timeout=10800
export random_state=42 
export problem="candle_attn"
export sync_val=0
export search="DMBS"
#!!! CONFIGURATION - END

# activate Python environment
source ../../../build/activate-dhenv.sh
export PYTHONPATH=../../../build/dhenv/lib/python3.8/site-packages/:$PYTHONPATH

# start MPS daemon on each node
./launch-mps-service.sh

# For MPS Client Application
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps 
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

if [[ "$sync_val" -eq 0 ]];
then
  export sync_str="async"
else
  export sync_str="sync"
fi

# AMBS
export log_dir="output/$problem-$search-$sync_str-$acq_func-$strategy-$COBALT_JOBSIZE-$RANKS_PER_NODE-$timeout-$random_state";

# profile gpu
mkdir $log_dir;
gpustat --no-color -i >> "$log_dir/gpustat.txt" &

echo "mpirun -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH -np $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) --hostfile $COBALT_NODEFILE python -m scalbo.exp --problem $problem \
--search $search \
--sync $sync_val \
--timeout $timeout \
--acq-func $acq_func \
--strategy $strategy \
--random-state $random_state \
--log-dir $log_dir \
--verbose 1";

mpirun -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH -np $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) --hostfile $COBALT_NODEFILE python -m scalbo.exp --problem $problem \
    --search $search \
    --sync $sync_val \
    --timeout $timeout \
    --acq-func $acq_func \
    --strategy $strategy \
    --random-state $random_state \
    --log-dir $log_dir \
    --verbose 1 

killall gpustat