#!/bin/bash
#PBS -l select=40:system=polaris
#PBS -l place=scatter
#PBS -l walltime=03:10:00
#PBS -q prod
#PBS -A datascience
#PBS -l filesystems=grand:home

set -xe

cd ${PBS_O_WORKDIR}


source ../../../build/activate-dhenv.sh

#!!! CONFIGURATION - START
export problem="dhb_combo"
export search="DBO"
export model="RF"
export acq_func="EI"
export scheduler_periode=80
export scheduler_rate=0.25
export pruning_strategy="SHA"
export objective_scaler="minmaxlog"
export timeout=10800
export random_state=42
#!!! CONFIGURATION - END

export NDEPTH=16
export NRANKS_PER_NODE=4
export NNODES=`wc -l < $PBS_NODEFILE`
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE ))
export OMP_NUM_THREADS=$NDEPTH


export log_dir="output/$problem-$search-$model-$acq_func-$pruning_strategy-$NNODES-$timeout-$random_state"
mkdir -p $log_dir

# Setup Redis Database
pushd $log_dir
redis-server $REDIS_CONF &
export DEEPHYPER_DB_HOST=$HOST
popd

sleep 5

# export GPUSTAT_LOG_DIR=$PBS_O_WORKDIR/$log_dir
# mpiexec -n ${NNODES} --ppn 1 --depth=1 --cpu-bind depth --envall ../profile_gpu_polaris.sh &

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
    --depth=${NDEPTH} \
    --cpu-bind depth \
    --envall \
    ../set_affinity_gpu_polaris.sh python -m scalbo.exp --problem $problem \
    --search $search \
    --model $model \
    --acq-func $acq_func \
    --objective-scaler $objective_scaler \
    --scheduler 1 \
    --scheduler-periode $scheduler_periode \
    --scheduler-rate $scheduler_rate \
    --random-state $random_state \
    --log-dir $log_dir \
    --pruning-strategy $pruning_strategy \
    --timeout $timeout \
    --max-steps 50 \
    --interval-steps 1 \
    --filter-duplicated 1
