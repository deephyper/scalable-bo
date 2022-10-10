#!/bin/bash
#PBS -l select=128:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:05:00
#PBS -q prod
#PBS -A datascience

cd ${PBS_O_WORKDIR}

source ../../../build/activate-dhenv.sh

#!!! CONFIGURATION - START
export RANKS_PER_NODE=4
export NNODES=`wc -l < $PBS_NODEFILE`
export acq_func="UCB"
export strategy="cl_max"
export model="RF"
export timeout=3600
export random_state=42
export problem="candle_combo"
export sync_val=0
export search="CBO"
#!!! CONFIGURATION - END

NTOTRANKS=$(( $NNODES * $RANKS_PER_NODE ))

# Number of CPU cores per rank
# 32 CPU cores per node - 1/4 for each rank
NDEPTH=8

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
export log_dir="output/$problem-$search-$sync_str-$model-$acq_func-$strategy-$NNODES-$RANKS_PER_NODE-$timeout-$random_state";

# profile gpu
mkdir $log_dir;
gpustat --no-color -i >> "$log_dir/gpustat.txt" &

echo "mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth \
    --envlist "LD_LIBRARY_PATH,PYTHONPATH,PATH" \
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

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth \
    --envlist "LD_LIBRARY_PATH,PYTHONPATH,PATH" \
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