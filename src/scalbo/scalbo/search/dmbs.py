from genericpath import exists
import logging
import pathlib
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time

import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"

from deephyper.search.hps import DMBSMPI

from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init_thread()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def execute(
    problem, sync, liar_strategy, timeout, max_evals, random_state, log_dir, cache_dir
):
    print(f"Starting the execution : {time.time()}")
    hp_problem = problem.hp_problem
    run = problem.run

    # define where the outputs are saved live (in cache-dir if possible)
    if cache_dir is not None and os.path.exists(cache_dir):
        search_cache_dir = os.path.join(cache_dir, "search")
        pathlib.Path(search_cache_dir).mkdir(parents=False, exist_ok=True)
        search_log_dir = search_cache_dir
    else:
        search_log_dir = log_dir

    pathlib.Path(search_log_dir).mkdir(parents=False, exist_ok=True)

    path_log_file = os.path.join(search_log_dir, f"deephyper.{rank}.log")
    logging.basicConfig(
        filename=path_log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
    )

    logging.info("Creation of the search instance...")
    search = DMBSMPI(
        hp_problem,
        run,
        sync_communication=sync,
        n_jobs=8,  # TODO: to be given according to the number of available hardware threads
        lazy_socket_allocation=False,
        log_dir=search_log_dir,
        random_state=random_state,
    )  # sampling boltzmann!
    logging.info("Creation of the search done")

    results = None
    logging.info("Starting the search...")
    print(f"Starting the search : {time.time()}")
    if rank == 0:
        results = search.search(timeout=timeout)
    else:
        search.search(timeout=timeout)
    logging.info("Search is done")

    print(f"Starting the report : {time.time()}")
    if rank == 0:
        results.to_csv(os.path.join(search_log_dir, f"results.csv"))

    if log_dir != search_log_dir:

        if rank == 0:
            os.system(f"mv {search_log_dir}/results.csv {log_dir}")
        comm.Barrier()

        if os.path.exists(search_log_dir):
            os.system(
                f"mv {path_log_file} {os.path.join(log_dir, f'deephyper.{rank}.log')}"
            )
