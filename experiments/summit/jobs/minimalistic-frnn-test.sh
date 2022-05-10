#!/bin/bash
#BSUB -nnodes 1
#BSUB -W 2:20
#BSUB -q debug
#BSUB -P fus145

# https://docs.olcf.ornl.gov/systems/summit_user_guide.html#common-bsub-options

source ../../../build/activate-dhenv.sh

export RANKS_PER_NODE=1
export NUM_NODES=1
export PYTHONPATH=../../../build/dhenv/lib/python3.7/site-packages/:$PYTHONPATH

# man jsrun
#        · -E  ,  –env  var=value  Exports the specified environment variable to the started processes before the
#   program is run.  This option overrides any locally defined environment variable with  the  same  name.
#   Existing  environment  variables can be propagated or new variables can be exported with corresponding
#   values.  For example, jsrun -E OMP_NUM_THREADS=4 -E PROG_INPUT.

# · -F , –env_eval var=val Evaluates the environment variable in the remote environment and sets the envi‐
# ronment variable in the environment of the started processes.

echo "Running: jsrun -E LD_LIBRARY_PATH -E PYTHONPATH -E PATH -n $(( $NUM_NODES * $RANKS_PER_NODE )) python -m scalbo.benchmark.minimalistic_frnn"
jsrun -E LD_LIBRARY_PATH -E PYTHONPATH -E PATH -n $(( $NUM_NODES * $RANKS_PER_NODE )) python -m scalbo.benchmark.minimalistic_frnn
