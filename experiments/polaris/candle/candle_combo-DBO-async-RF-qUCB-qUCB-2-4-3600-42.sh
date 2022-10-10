#!/bin/bash
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -q debug-scaling 
#PBS -A datascience

cd ${PBS_O_WORKDIR}

source ../../../build/activate-dhenv.sh

#!!! CONFIGURATION - START
export NRANKS_PER_NODE=4
export NNODES=`wc -l < $PBS_NODEFILE`
export acq_func="qUCB"
export strategy="qUCB"
export model="RF"
export timeout=3600
export random_state=42
export problem="candle_combo"
export sync_val=0
export search="DBO"
#!!! CONFIGURATION - END

NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE ))

# Number of CPU cores per rank
# 32 CPU cores per node - 1/4 for each rank
NDEPTH=7

# activate Python environment
source ../../../build/activate-dhenv.sh
export PYTHONPATH=../../../build/dhenv/lib/python3.8/site-packages/:$PYTHONPATH

if [[ "$sync_val" -eq 0 ]];
then
  export sync_str="async"
else
  export sync_str="sync"
fi

# CBO
export log_dir="output/$problem-$search-$sync_str-$model-$acq_func-$strategy-$NNODES-$NRANKS_PER_NODE-$timeout-$random_state";

# profile gpu
mkdir $log_dir;
gpustat --no-color -i >> "$log_dir/gpustat.txt" &

echo "mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
    --envall \  
    ./set_affinity_gpu_polaris.sh python -m scalbo.exp --problem $problem \
    python -m scalbo.exp --problem $problem \
    --search $search \
    --model $model \
    --sync $sync_val \
    --timeout $timeout \
    --acq-func $acq_func \
    --strategy $strategy \
    --random-state $random_state \
    --log-dir $log_dir \
    --verbose 1";

export MPICH_ASYNC_PROGRESS=1

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
    --envall \
    ./set_affinity_gpu_polaris.sh python -m scalbo.exp --problem $problem \
    --search $search \
    --model $model \
    --sync $sync_val \
    --timeout $timeout \
    --acq-func $acq_func \
    --strategy $strategy \
    --random-state $random_state \
    --log-dir $log_dir \
    --verbose 1 