#!/bin/bash
#PBS -l select=4:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -q debug-scaling 
#PBS -A datascience
#PBS -l filesystems=grand:home

set -x

cd ${PBS_O_WORKDIR}

source ../../../build/activate-dhenv.sh

#!!! CONFIGURATION - START
export NRANKS_PER_NODE=4
export acq_func="qUCB"
export strategy="qUCB"
export model="MF"
export timeout=3600
export random_state=42
export problem="dhb_combo"
export sync_val=0
export search="DBOS4M"
export MPICH_ASYNC_PROGRESS=0
export MPICH_CH4_OFI_ENABLE_DATA_AUTO_PROGRESS=0
export NDEPTH=4
#!!! CONFIGURATION - END

export NNODES=`wc -l < $PBS_NODEFILE`
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE ))
export OMP_NUM_THREADS=$NDEPTH

if [[ "$sync_val" -eq 0 ]];
then
  export sync_str="async"
else
  export sync_str="sync"
fi

# CBO
export log_dir="output/$problem-$search-$sync_str-$model-$acq_func-$strategy-$NNODES-$NRANKS_PER_NODE-$timeout-$random_state"
mkdir $log_dir

export GPUSTAT_LOG_DIR=$PBS_O_WORKDIR/$log_dir
mpiexec -n ${NNODES} --ppn 1 --depth=1 --cpu-bind depth --envall ../profile_gpu_polaris.sh &

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
    --depth=${NDEPTH} \
    --cpu-bind depth \
    --envall \
    ../set_affinity_gpu_polaris.sh python -m scalbo.exp --problem $problem \
    --search $search \
    --model $model \
    --sync $sync_val \
    --timeout $timeout \
    --acq-func $acq_func \
    --strategy $strategy \
    --random-state $random_state \
    --log-dir $log_dir \
    --verbose 1 
