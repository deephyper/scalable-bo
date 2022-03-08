#!/bin/bash

qsub ackley-ambs-sync-boltzmann-128-1-1800-42.sh
qsub ackley-ambs-async-boltzmann-128-8-1800-42.sh
qsub ackley-dmbs-sync-boltzmann-128-8-1800-42.sh
qsub ackley-dmbs-async-boltzmann-128-8-1800-42.sh
qsub ackley-ambs-async-boltzmann-128-16-1800-42.sh
qsub ackley-dmbs-sync-boltzmann-128-16-1800-42.sh
qsub ackley-dmbs-async-boltzmann-128-16-1800-42.sh
qsub ackley-ambs-async-boltzmann-128-32-1800-42.sh
qsub ackley-dmbs-sync-boltzmann-128-32-1800-42.sh
qsub ackley-dmbs-async-boltzmann-128-32-1800-42.sh