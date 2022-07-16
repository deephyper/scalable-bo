#!/bin/bash
#BSUB -nnodes 1
#BSUB -W 2:00
#BSUB -q debug
#BSUB -P fus145

# https://docs.olcf.ornl.gov/systems/summit_user_guide.html#common-bsub-options

source ../../../build/activate-dhenv.sh

export RANKS_PER_NODE=1
export NUM_NODES=1
export PYTHONPATH=../../../build/dhenv/lib/python3.8/site-packages/:$PYTHONPATH

which python
echo $PATH

echo "Running: jsrun -E LD_LIBRARY_PATH -E PYTHONPATH -E PATH -n $(( $NUM_NODES * $RANKS_PER_NODE )) python -m scalbo.benchmark.minimalistic_frnn"
jsrun -E LD_LIBRARY_PATH -E PYTHONPATH -E PATH -n $(( $NUM_NODES * $RANKS_PER_NODE )) -r1 -g6 -a1 -c42   -bpacked:42 which python
jsrun -E LD_LIBRARY_PATH -E PYTHONPATH -E PATH -n $(( $NUM_NODES * $RANKS_PER_NODE )) -r1 -g6 -a1 -c42   -bpacked:42 printenv
jsrun -E LD_LIBRARY_PATH -E PYTHONPATH -E PATH -n $(( $NUM_NODES * $RANKS_PER_NODE )) -r1 -g6 -a1 -c42     -bpacked:42 python -c "import scalbo; print(scalbo.__path__)"
jsrun -E LD_LIBRARY_PATH -E PYTHONPATH -E PATH -n $(( $NUM_NODES * $RANKS_PER_NODE )) -r1 -g6 -a1 -c42 -bpacked:42 python -m scalbo.benchmark.minimalistic_frnn
