#!/bin/bash
#COBALT -n 1
#COBALT -t 20
#COBALT -q single-gpu
#COBALT -A datascience

# DOC: https://docs.nvidia.com/deploy/mps/#topic_6_1


# Starting MPS Control Daemon
export CUDA_VISIBLE_DEVICES=0 # Select the GPU
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d # Start the daemon.

# Starting MPS Client Application
unset CUDA_VISIBLE_DEVICES
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps # Set to the same location as the MPS control daemon
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log # Set to the same location as the MPS control daemon