#!/bin/bash
#PBS -l select=10:system=polaris
#PBS -l place=scatter
#PBS -l walltime=03:00:00
#PBS -q prod
#PBS -A datascience
#PBS -l filesystems=grand:home

set -xe

cd ${PBS_O_WORKDIR}

source ../../../build/activate-dhenv.sh

#!!! CONFIGURATION - START
export problem="dhb_combo"
export OPTUNA_N_OBJECTIVES=3
export search="OPT-TPE"
export pruning_strategy="NONE"
export timeout=10200
export random_state=42
export lower_bounds="0.85,None,None"
#!!! CONFIGURATION - END

export DEEPHYPER_BENCHMARK_MOO="1"

export NDEPTH=16
export NRANKS_PER_NODE=4
export NNODES=`wc -l < $PBS_NODEFILE`
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE ))
export OMP_NUM_THREADS=$NDEPTH


export log_dir="output/$problem-$search-$NNODES-$timeout-$random_state-MOO"
mkdir -p $log_dir

### Setup Postgresql Database - START ###
export OPTUNA_DB_DIR="$log_dir/optunadb"
export OPTUNA_DB_HOST=$HOST
initdb -D "$OPTUNA_DB_DIR"

# Set authentication policy to "trust" for all users
echo "host    all             all             .hsn.cm.polaris.alcf.anl.gov               trust" >> "$OPTUNA_DB_DIR/pg_hba.conf"

# Set the limit of max connections to 2048
sed -i "s/max_connections = 100/max_connections = 2048/" "$OPTUNA_DB_DIR/postgresql.conf"

# start the server in the background and listen to all interfaces
pg_ctl -D $OPTUNA_DB_DIR -l "$log_dir/db.log" -o "-c listen_addresses='*'" start

createdb hpo
### Setup Postgresql Database - END ###

sleep 5

# export GPUSTAT_LOG_DIR=$PBS_O_WORKDIR/$log_dir
# mpiexec -n ${NNODES} --ppn 1 --depth=1 --cpu-bind depth --envall ../profile_gpu_polaris.sh &

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} \
    --depth=${NDEPTH} \
    --cpu-bind depth \
    --envall \
    ../set_affinity_gpu_polaris.sh python -m scalbo.exp --problem $problem \
    --search $search \
    --random-state $random_state \
    --log-dir $log_dir \
    --pruning-strategy $pruning_strategy \
    --timeout $timeout \
    --max-steps 50 \
    --lower-bounds $lower_bounds

dropdb hpo
pg_ctl -D $OPTUNA_DB_DIR -l "$log_dir/db.log" stop