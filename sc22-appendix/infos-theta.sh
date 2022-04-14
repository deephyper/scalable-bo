#!/bin/bash
#COBALT -n 1
#COBALT -t 10
#COBALT -q debug-cache-quad
#COBALT -A datascience

source ../build/activate-dhenv.sh

cat /etc/os-release &> theta_os.out
./collect_environment.sh &> theta_env.out