#!/bin/bash
#PBS -l select=160:system=polaris
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
export model="ET"
export acq_func="UCBd"
export scheduler_periode=48
export scheduler_rate=0.1
export pruning_strategy="NONE"
export objective_scaler="quantile-uniform"
export timeout=10200
export random_state=42
export scalar_func="Linear"
#export lower_bounds="0.85,None,None"
export acq_func_optimizer="mixedga"
#!!! CONFIGURATION - END

export DEEPHYPER_BENCHMARK_MOO="1"

export NDEPTH=16
export NRANKS_PER_NODE=4
export NNODES=`wc -l < $PBS_NODEFILE`
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE ))
export OMP_NUM_THREADS=$NDEPTH


export log_dir="output/$problem-$search-$model-$acq_func-$acq_func_optimizer-$NNODES-$timeout-$random_state-MOO-noconst"
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
    --filter-duplicated 1 \
    --scalar-func $scalar_func \
    --acq-func-optimizer $acq_func_optimizer #\
    #--lower-bounds $lower_bounds
