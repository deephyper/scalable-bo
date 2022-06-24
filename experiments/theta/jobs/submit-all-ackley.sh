#!/bin/bash

#* Asynchronous

# Centralized Liar Strategy #! OK
# qsub ackley-CBO-async-UCB-cl_max-1-1-1800-42.sh
# qsub ackley-CBO-async-UCB-cl_max-128-1-1800-42.sh

# Centralized Boltzmann #! OK
# qsub ackley-CBO-async-UCB-boltzmann-128-1-1800-42.sh
# qsub ackley-CBO-async-UCB-boltzmann-128-2-1800-42.sh
# qsub ackley-CBO-async-UCB-boltzmann-128-4-1800-42.sh
# qsub ackley-CBO-async-UCB-boltzmann-128-8-1800-42.sh
# qsub ackley-CBO-async-UCB-boltzmann-128-16-1800-42.sh
# qsub ackley-CBO-async-UCB-boltzmann-128-32-1800-42.sh

# Centralized qUCB #! READY
qsub ackley-CBO-async-qUCB-qUCB-128-1-1800-42.sh
qsub ackley-CBO-async-qUCB-qUCB-128-2-1800-42.sh
qsub ackley-CBO-async-qUCB-qUCB-128-4-1800-42.sh
qsub ackley-CBO-async-qUCB-qUCB-128-8-1800-42.sh
qsub ackley-CBO-async-qUCB-qUCB-128-16-1800-42.sh
qsub ackley-CBO-async-qUCB-qUCB-128-32-1800-42.sh

# Distributed Boltzmann #! OK
# qsub ackley-DBO-async-UCB-boltzmann-128-1-1800-42.sh
# qsub ackley-DBO-async-UCB-boltzmann-128-2-1800-42.sh
# qsub ackley-DBO-async-UCB-boltzmann-128-4-1800-42.sh
# qsub ackley-DBO-async-UCB-boltzmann-128-8-1800-42.sh
# qsub ackley-DBO-async-UCB-boltzmann-128-16-1800-42.sh
# qsub ackley-DBO-async-UCB-boltzmann-128-32-1800-42.sh

# Distributed qUCB #! OK
# qsub ackley-DBO-async-qUCB-qUCB-128-1-1800-42.sh
# qsub ackley-DBO-async-qUCB-qUCB-128-2-1800-42.sh
# qsub ackley-DBO-async-qUCB-qUCB-128-4-1800-42.sh
# qsub ackley-DBO-async-qUCB-qUCB-128-8-1800-42.sh
# qsub ackley-DBO-async-qUCB-qUCB-128-16-1800-42.sh
# qsub ackley-DBO-async-qUCB-qUCB-128-32-1800-42.sh

#* Synchronous

# Centralized Liar Strategy #! READY
qsub ackley-CBO-sync-UCB-cl_max-128-1-1800-42.sh

# Centralized Boltzmann #! OK
# qsub ackley-CBO-sync-UCB-boltzmann-128-1-1800-42.sh

# Centralized qUCB #! READY
qsub ackley-CBO-sync-qUCB-qUCB-128-1-1800-42.sh

# Distributed Boltzmann #! OK
# qsub ackley-DBO-sync-UCB-boltzmann-128-1-1800-42.sh

# Distributed qUCB #! READY
qsub ackley-DBO-sync-qUCB-qUCB-128-1-1800-42.sh