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

from deephyper.search.hps import MPIDistributedBO
from deephyper.stopper import (
    SuccessiveHalvingStopper,
    MedianStopper,
    LCModelStopper,
    IdleStopper,
    ConstantStopper,
)

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
    timeout,
    max_evals,
    random_state,
    log_dir,
    cache_dir,
    n_jobs,
    model,
    pruning_strategy=None,
    scheduler=False,
    scheduler_periode=25,
    scheduler_rate=0.1,
    filter_duplicated=False,
    objective_scaler="identity",
    max_steps=None,
    interval_steps=1,
    scalar_func="Chebyshev",
    lower_bounds=None,
    acq_func_optimizer="sampling",
    **kwargs,
):
    # define where the outputs are saved live (in cache-dir if possible)
    if cache_dir is not None and os.path.exists(cache_dir):
        search_log_dir = os.path.join(cache_dir, "search")
    else:
        search_log_dir = log_dir

    pathlib.Path(search_log_dir).mkdir(parents=False, exist_ok=True)

    path_log_file = os.path.join(search_log_dir, f"deephyper.{rank}.log")
    if rank == 0:
        logging.basicConfig(
            filename=path_log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
            force=True,
        )

    rs = np.random.RandomState(random_state)

    hp_problem = problem.hp_problem
    run = problem.run

    evaluator = MPIDistributedBO.bootstrap_evaluator(
        run,
        evaluator_type="serial",  # one worker to evaluate the run-function per rank
        storage_type="redis",
        storage_kwargs={
            "host": os.environ["DEEPHYPER_DB_HOST"],
            "port": 6379,
        },
        comm=comm,
        root=0,
    )

    if pruning_strategy:


        if pruning_strategy == "SHA":
            stopper = SuccessiveHalvingStopper(
                max_steps=max_steps,
                min_steps=1,
                min_fully_completed=1,
                reduction_factor=3,  # 4 is the default param. value in Optuna
            )
        elif pruning_strategy == "MED":
            stopper = MedianStopper(
                max_steps=max_steps,
                min_steps=1,
                min_fully_completed=1,
                interval_steps=interval_steps,
            )
        elif pruning_strategy == "NONE":
            stopper = IdleStopper(max_steps=max_steps)
        elif pruning_strategy[:5] == "CONST":  # CONST4
            stop_step = int(pruning_strategy[5:])
            stopper = ConstantStopper(max_steps=max_steps, stop_step=stop_step)
        else:
            stopper = LCModelStopper(
                max_steps=max_steps,
                lc_model=pruning_strategy.lower(),
            )
    else:
        stopper = None

    logging.info("Creation of the search instance...")

    if scheduler:
        scheduler = {
            "type": "periodic-exp-decay",
            "periode": scheduler_periode,
            "rate": scheduler_rate,
        }
    else:
        scheduler = None

    if lower_bounds is not None:
        lower_bounds = [float(lb) if lb != "None" else None for lb in lower_bounds.split(",")]

    search = MPIDistributedBO(
        hp_problem,
        evaluator,
        n_jobs=n_jobs,
        log_dir=search_log_dir,
        random_state=rs,
        acq_func=acq_func,
        acq_optimizer=acq_func_optimizer,
        acq_optimizer_freq=1,
        surrogate_model=model,
        filter_duplicated=filter_duplicated,
        filter_failures="min",
        scheduler=scheduler,
        objective_scaler=objective_scaler,
        stopper=stopper,
        moo_scalarization_strategy=scalar_func,
        moo_lower_bounds=lower_bounds,
        verbose=0,
    )
    logging.info("Creation of the search done")

    results = None
    logging.info("Starting the search...")
    if rank == 0:
        results = search.search(max_evals=max_evals, timeout=timeout)
    else:
        search.search(max_evals=max_evals, timeout=timeout)
    logging.info("Search is done")

    if log_dir != search_log_dir:

        if rank == 0:
            os.system(f"mv {search_log_dir}/results.csv {log_dir}")

            os.system(f"mv {path_log_file} {log_dir}")

    comm.Barrier()
    comm.Abort()
