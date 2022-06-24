#!/bin/bash

qsub ackley-DBO-async-qUCB-qUCB-128-32-1800-42.sh
qsub ackley-DBO-sync-UCB-boltzmann-128-32-1800-42.sh

qsub hartmann6D-DBO-async-qUCB-qUCB-128-32-1800-42.sh
qsub hartmann6D-DBO-sync-UCB-boltzmann-128-32-1800-42.sh

qsub levy-DBO-async-qUCB-qUCB-128-32-1800-42.sh
qsub levy-DBO-sync-UCB-boltzmann-128-32-1800-42.sh

qsub griewank-DBO-async-qUCB-qUCB-128-32-1800-42.sh
qsub griewank-DBO-sync-UCB-boltzmann-128-32-1800-42.sh

qsub schwefel-DBO-async-qUCB-qUCB-128-32-1800-42.sh
qsub schwefel-DBO-sync-UCB-boltzmann-128-32-1800-42.sh