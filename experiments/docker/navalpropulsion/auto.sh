#!/bin/bash

export NUM_WORKERS=8
export DEEPHYPER_BENCHMARK_SIMULATE_RUN_TIME=1 
export DEEPHYPER_BENCHMARK_PROP_REAL_RUN_TIME=0.0001

./dhb_navalpropulsion-CBO-DUMMY-UCB.sh
./dhb_navalpropulsion-CBO-RF-UCB.sh
./dhb_navalpropulsion-CBO-RF-UCB-SHA.sh
./dhb_navalpropulsion-DBO-RF-UCB.sh
./dhb_navalpropulsion-DBO-RF-UCB-SHA.sh