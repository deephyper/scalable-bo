#!/bin/bash
#COBALT -n 1
#COBALT -t 10
#COBALT -q full-node
#COBALT -A datascience

source ../build/activate-dhenv.sh

cat /etc/os-release &> thetagpu_os.out
./collect_environment.sh &> thetagpu_env.out