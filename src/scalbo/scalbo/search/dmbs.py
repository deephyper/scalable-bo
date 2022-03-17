import logging
import pathlib
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"
mpi4py.rc.recv_mprobe = False

import numpy as np

from deephyper.search.hps import DMBSMPI

from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init_thread()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def execute(
    problem,
    sync,
    acq_func,
    strategy,
    timeout,
    random_state,
    log_dir,
    cache_dir,
):

    # define where the outputs are saved live (in cache-dir if possible)
    if cache_dir is not None and os.path.exists(cache_dir):
        search_log_dir = os.path.join(cache_dir, "search")
    else:
        search_log_dir = log_dir

    pathlib.Path(search_log_dir).mkdir(parents=False, exist_ok=True)

    if rank == 0:
        path_log_file = os.path.join(search_log_dir, f"deephyper.{rank}.log")
        logging.basicConfig(
            filename=path_log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
            force=True,
        )

    rs = np.random.RandomState(random_state)
    # rank_seed = rs.randint(low=0, high=2**32, size=size)[rank]

    hp_problem = problem.hp_problem
    run = problem.run

    logging.info("Creation of the search instance...")

    search = DMBSMPI(
        hp_problem,
        run,
        sync_communication=sync,
        sync_communication_freq=1,
        n_jobs=4,  # TODO: to be given according to the number of available hardware threads
        lazy_socket_allocation=False,
        log_dir=search_log_dir,
        random_state=rs,
        acq_func=acq_func,
    )  # sampling boltzmann!
    logging.info("Creation of the search done")

    results = None
    logging.info("Starting the search...")
    if rank == 0:
        results = search.search(timeout=timeout)
    else:
        search.search(timeout=timeout)
    logging.info("Search is done")

    # if rank == 0:
    #     results.to_csv(os.path.join(search_log_dir, "results.csv"))

    if log_dir != search_log_dir:

        if rank == 0:
            os.system(f"mv {search_log_dir}/results.csv {log_dir}")

            os.system(f"mv {path_log_file} {log_dir}")
