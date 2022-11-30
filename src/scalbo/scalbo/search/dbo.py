import getpass
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

from deephyper.search.hps import DBO
from deephyper.evaluator import distributed, SerialEvaluator
from deephyper.evaluator.storage import RedisStorage
from deephyper.stopper import SuccessiveHalvingStopper, MedianStopper

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
    distributed_backend="mpi",
    pruning_strategy=None,
    scheduler=False,
    scheduler_periode=25,
    scheduler_rate=0.1,
    filter_duplicated=False,
    objective_scaler="identity",
    **kwargs,
):

    # define where the outputs are saved live (in cache-dir if possible)
    if cache_dir is not None and os.path.exists(cache_dir):
        search_log_dir = os.path.join(cache_dir, "search")
    else:
        search_log_dir = log_dir

    pathlib.Path(search_log_dir).mkdir(parents=False, exist_ok=True)

    path_log_file = os.path.join(search_log_dir, f"deephyper.{rank}.log")
    logging.basicConfig(
        filename=path_log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
        force=True,
    )

    rs = np.random.RandomState(random_state)

    hp_problem = problem.hp_problem
    run = problem.run

    DistributedEvaluator = distributed(backend=distributed_backend)(SerialEvaluator)

    storage = RedisStorage()
    storage.connect()
    evaluator = DistributedEvaluator(run, storage=storage)
    if pruning_strategy:

        # Optuna Pruner
        # username = getpass.getuser()
        # host = os.environ["OPTUNA_DB_HOST"]

        # storage = f"postgresql://{username}@{host}:5432/hpo"

        if pruning_strategy == "SHA":
            stopper = SuccessiveHalvingStopper(
                min_fully_completed=5, min_budget=3, reduction_factor=3
            )
        elif pruning_strategy == "MED":
            stopper = MedianStopper(
                min_fully_completed=5, min_budget=30, interval_steps=10
            )
        elif pruning_strategy == "NONE":
            stopper = None
        else:
            raise ValueError(f"Wrong pruning strategy '{pruning_strategy}'")
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

    search = DBO(
        hp_problem,
        evaluator,
        sync_communication=sync,
        n_jobs=n_jobs,
        log_dir=search_log_dir,
        random_state=rs,
        acq_func=acq_func,
        surrogate_model=model,
        filter_duplicated=filter_duplicated,
        scheduler=scheduler,
        objective_scaler=objective_scaler,
        stopper=stopper,
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
