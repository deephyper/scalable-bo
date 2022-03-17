#!/bin/bash

#* Asynchronous

# Centralized Liar Strategy #! OK
# qsub ackley-AMBS-async-UCB-cl_max-1-1-1800-42.sh
# qsub ackley-AMBS-async-UCB-cl_max-128-1-1800-42.sh

# Centralized Boltzmann #! OK
# qsub ackley-AMBS-async-UCB-boltzmann-128-1-1800-42.sh
# qsub ackley-AMBS-async-UCB-boltzmann-128-2-1800-42.sh
# qsub ackley-AMBS-async-UCB-boltzmann-128-4-1800-42.sh
# qsub ackley-AMBS-async-UCB-boltzmann-128-8-1800-42.sh
# qsub ackley-AMBS-async-UCB-boltzmann-128-16-1800-42.sh
# qsub ackley-AMBS-async-UCB-boltzmann-128-32-1800-42.sh

# Centralized qUCB #! READY
qsub ackley-AMBS-async-qUCB-qUCB-128-1-1800-42.sh
qsub ackley-AMBS-async-qUCB-qUCB-128-2-1800-42.sh
qsub ackley-AMBS-async-qUCB-qUCB-128-4-1800-42.sh
qsub ackley-AMBS-async-qUCB-qUCB-128-8-1800-42.sh
qsub ackley-AMBS-async-qUCB-qUCB-128-16-1800-42.sh
qsub ackley-AMBS-async-qUCB-qUCB-128-32-1800-42.sh

# Distributed Boltzmann #! OK
# qsub ackley-DMBS-async-UCB-boltzmann-128-1-1800-42.sh
# qsub ackley-DMBS-async-UCB-boltzmann-128-2-1800-42.sh
# qsub ackley-DMBS-async-UCB-boltzmann-128-4-1800-42.sh
# qsub ackley-DMBS-async-UCB-boltzmann-128-8-1800-42.sh
# qsub ackley-DMBS-async-UCB-boltzmann-128-16-1800-42.sh
# qsub ackley-DMBS-async-UCB-boltzmann-128-32-1800-42.sh

# Distributed qUCB #! OK
# qsub ackley-DMBS-async-qUCB-qUCB-128-1-1800-42.sh
# qsub ackley-DMBS-async-qUCB-qUCB-128-2-1800-42.sh
# qsub ackley-DMBS-async-qUCB-qUCB-128-4-1800-42.sh
# qsub ackley-DMBS-async-qUCB-qUCB-128-8-1800-42.sh
# qsub ackley-DMBS-async-qUCB-qUCB-128-16-1800-42.sh
# qsub ackley-DMBS-async-qUCB-qUCB-128-32-1800-42.sh

#* Synchronous

# Centralized Liar Strategy #! READY
qsub ackley-AMBS-sync-UCB-cl_max-128-1-1800-42.sh

# Centralized Boltzmann #! OK
# qsub ackley-AMBS-sync-UCB-boltzmann-128-1-1800-42.sh

# Centralized qUCB #! READY
qsub ackley-AMBS-sync-qUCB-qUCB-128-1-1800-42.sh

# Distributed Boltzmann #! OK
# qsub ackley-DMBS-sync-UCB-boltzmann-128-1-1800-42.sh

# Distributed qUCB #! READY
qsub ackley-DMBS-sync-qUCB-qUCB-128-1-1800-42.sh