#!/bin/bash

# DOC: https://docs.nvidia.com/deploy/mps/#topic_6_1


# Starting MPS Control Daemon
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # Select the GPU
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

# Start the daemon.
mpirun -x LD_LIBRARY_PATH -x PATH \
    -np $COBALT_JOBSIZE \
    --hostfile $COBALT_NODEFILE \
    echo quit | nvidia-cuda-mps-control