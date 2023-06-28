#!/bin/bash
#PBS -l select=480
#PBS -l walltime=03:10:00
#PBS -q prod
#PBS -l filesystems=home:grand
#PBS -A datascience

set -xe

cd ${PBS_O_WORKDIR}


source ../../../build/activate-dhenv.sh

#!!! CONFIGURATION - START
export problem="dhb_combo"
export search="OPT-TPE"
export pruning_strategy="SHA"
export timeout=10800
export random_state=42
#!!! CONFIGURATION - END

export NRANKS_PER_NODE=4
export NDEPTH=$((64 / $NRANKS_PER_NODE))
export NNODES=`wc -l < $PBS_NODEFILE`
export NTOTRANKS=$(( $NNODES * $NRANKS_PER_NODE ))
export OMP_NUM_THREADS=$NDEPTH

export log_dir="output/$problem-$search-$pruning_strategy-$NNODES-$timeout-$random_state"
mkdir -p $log_dir

### Setup Postgresql Database - START ###
export OPTUNA_DB_DIR="$log_dir/optunadb"
export OPTUNA_DB_HOST=$HOST
initdb -D "$OPTUNA_DB_DIR"

# Set authentication policy to "trust" for all users
echo "host    all             all             .hsn.cm.polaris.alcf.anl.gov               trust" >> "$OPTUNA_DB_DIR/pg_hba.conf"

# Set the limit of max connections to 2048
sed -i "s/max_connections = 100/max_connections = 4096/" "$OPTUNA_DB_DIR/postgresql.conf"

# start the server in the background and listen to all interfaces
pg_ctl -D $OPTUNA_DB_DIR -l "$log_dir/db.log" -o "-c listen_addresses='*'" start

createdb hpo
### Setup Postgresql Database - END ###

sleep 5

#export GPUSTAT_LOG_DIR=$PBS_O_WORKDIR/$log_dir
#mpiexec -n ${NNODES} --ppn 1 --depth=1 --cpu-bind depth --envall ../profile_gpu_polaris.sh &

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
    --max-steps 50 

dropdb hpo
pg_ctl -D $OPTUNA_DB_DIR -l "$log_dir/db.log" stop
