#!/bin/bash

qsub hartmann6D-DMBS-async-qUCB-qUCB-128-32-1800-42.sh
qsub hartmann6D-DMBS-sync-UCB-boltzmann-128-32-1800-42.sh

qsub levy-DMBS-async-qUCB-qUCB-128-32-1800-42.sh
qsub levy-DMBS-sync-UCB-boltzmann-128-32-1800-42.sh

qsub griewank-DMBS-async-qUCB-qUCB-128-32-1800-42.sh
qsub griewank-DMBS-sync-UCB-boltzmann-128-32-1800-42.sh

qsub schwefel-DMBS-async-qUCB-qUCB-128-32-1800-42.sh
qsub schwefel-DMBS-sync-UCB-boltzmann-128-32-1800-42.sh