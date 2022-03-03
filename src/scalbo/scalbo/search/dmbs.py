import logging
import pathlib
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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
    pathlib.Path(log_dir).mkdir(parents=False, exist_ok=True)

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
        n_jobs=4, # TODO: to be given according to the number of available hardware threads
        lazy_socket_allocation=False,
        sync_communication=sync,
    ) # sampling boltzmann!
    logging.info("Creation of the search done")

    results = None
    logging.info("Starting the search...")
    if rank == 0:
        results = search.search(timeout=timeout)
    else:
        search.search(timeout=timeout)
    logging.info("Search is done")

    
    if "/dev/shm" in log_dir:
        results_exp = os.path.join("results", exp_name)
        results_exp_logs = os.path.join(results_exp, "deephyper-logs")
        if rank == 0:
            pathlib.Path("results").mkdir(parents=False, exist_ok=True)
            pathlib.Path(results_exp).mkdir(parents=False, exist_ok=True)
            os.system(f"mv {log_dir}/* {results_exp}")
            pathlib.Path(results_exp_logs).mkdir(parents=False, exist_ok=True)
            os.system(f"mv {results_exp}/deephyper.*.log {results_exp_logs}")

        comm.Barrier()

        if rank > 0:
            if os.path.isfile(f"{log_dir}/deephyper.{rank}.log"):
                os.system(f"mv {log_dir}/deephyper.{rank}.log {results_exp_logs}")