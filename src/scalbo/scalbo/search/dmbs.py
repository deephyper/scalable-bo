import logging
import pathlib
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import time

import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"

import numpy as np
import scipy.stats
from deephyper.search.hps import DMBSMPI

from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init_thread()
    
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def execute(problem, sync, liar_strategy, timeout, max_evals, random_state, log_dir_spec):
    problem_name = problem.name
    hp_problem = problem.hp_problem
    run = problem.run
    config = [
        problem_name,
        str(size),
        "dmbs",
        "sync" if sync else "async",
        liar_strategy,
        str(timeout),
        "0" if max_evals == -1 else str(max_evals),
        str(random_state)
    ]
    exp_name = (
        "-".join(config)
    )

    log_dir = os.path.join(log_dir_spec, exp_name)
    pathlib.Path(log_dir).mkdir(parents=False, exist_ok=False)

    log_file = os.path.join(log_dir, f"deephyper.{rank}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
    )


    logging.info("Creation of the search instance...")
    search = DMBSMPI(
        hp_problem,
        run,
        log_dir=log_dir,
        n_jobs=4,
        lazy_socket_allocation=False,
        sync_communication=sync,
    ) # sampling boltzmann!
    logging.info("Creation of the search done")

    results = None
    logging.info("Starting the search...")
    if rank == 0:
        results = search.search(timeout=60*timeout)
    else:
        search.search(timeout=60*timeout)
    logging.info("Search is done")

    
    if "/dev/shm" in log_dir:
        results_path = os.path.join("results", exp_name)
        logs_path = os.path.join(results_path, "deephyper-logs")
        if rank == 0:
            pathlib.Path("results").mkdir(parents=False, exist_ok=True)
            pathlib.Path(results_path).mkdir(parents=False, exist_ok=True)
            os.system(f"mv {"log_dir/*"} {results_path}")
            pathlib.Path(logs_path).mkdir(parents=False, exist_ok=True)
            os.system(f"mv {results_path}/deephyper.*.log {logs_path}")

        comm.Barrier()

        if rank > 0 and rank % args.num_ranks_per_node == 0:
            os.system(f"mv {log_dir}/* {logs_path}")