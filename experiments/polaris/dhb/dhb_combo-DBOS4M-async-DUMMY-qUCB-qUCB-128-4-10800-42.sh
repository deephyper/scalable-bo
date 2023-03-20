#!/bin/bash
#PBS -l select=128:system=polaris
#PBS -l place=scatter
#PBS -l walltime=03:05:00
#PBS -q prod
#PBS -A datascience
#PBS -l filesystems=grand:home

set -x

cd ${PBS_O_WORKDIR}

source ../../../build/activate-dhenv.sh

#!!! CONFIGURATION - START
export NRANKS_PER_NODE=4
export acq_func="qUCB"
export strategy="qUCB"
export model="DUMMY"
export timeout=10800
export random_state=42
export problem="dhb_combo"
export sync_val=0
export search="DBOS4M"
export MPICH_ASYNC_PROGRESS=0
export MPICH_CH4_OFI_ENABLE_DATA_AUTO_PROGRESS=0
export NDEPTH=16
#!!! CONFIGURATION - END

export NNODES=`wc -l < $PBS_NODEFILE`
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE ))
export OMP_NUM_THREADS=$(expr $NDEPTH - 2)

if [[ "$sync_val" -eq 0 ]];
then
  export sync_str="async"
else
  export sync_str="sync"
fi

# CBO
export log_dir="output/$problem-$search-$sync_str-$model-$acq_func-$strategy-$NNODES-$NRANKS_PER_NODE-$timeout-$random_state"
mkdir -p $log_dir

# Create database
export OPTUNA_DB_DIR="$log_dir/optunadb"
export OPTUNA_DB_HOST=$HOST
initdb -D "$OPTUNA_DB_DIR"
echo "host    all             all             .hsn.cm.polaris.alcf.anl.gov               trust" >> "$OPTUNA_DB_DIR/pg_hba.conf"

# increase the limit of max connections
sed -i "s/max_connections = 100/max_connections = 2048/" "$OPTUNA_DB_DIR/postgresql.conf"

pg_ctl -D $OPTUNA_DB_DIR -l "$log_dir/db.log" -o "-c listen_addresses='*'" start
createdb hpo

sleep 5

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

dropdb hpo
pg_ctl -D $OPTUNA_DB_DIR -l "$log_dir/db.log" stop
