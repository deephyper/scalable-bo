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
    hp_problem = problem.hp_problem
    run = problem.run

    # define where the outputs are saved live (in cache-dir if possible)
    if cache_dir is not None and os.path.exists(cache_dir):
        search_log_dir = cache_dir
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
        log_dir=log_dir,
        n_jobs=1,  # TODO: to be given according to the number of available hardware threads
        lazy_socket_allocation=False,
        sync_communication=sync,
    )  # sampling boltzmann!
    logging.info("Creation of the search done")

    results = None
    logging.info("Starting the search...")
    if rank == 0:
        results = search.search(timeout=timeout)
    else:
        search.search(timeout=timeout)
    logging.info("Search is done")

    if rank == 0:
        results.to_csv(os.path.join(search_log_dir, f"results.csv"))

    if log_dir != search_log_dir:

        if rank == 0:
            os.system(f"mv {search_log_dir} {log_dir}")
        comm.Barrier()

        if os.path.exists(search_log_dir):
            os.system(
                f"mv {path_log_file} {os.path.join(log_dir, f'deephyper.{rank}.log')}"
            )
