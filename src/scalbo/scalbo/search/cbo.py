import logging
import pathlib
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"
# mpi4py.rc.recv_mprobe = False

import numpy as np

from deephyper.evaluator import Evaluator
from deephyper.evaluator.storage import Storage
from deephyper.search.hps import CBO
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
    strategy,
    timeout,
    max_evals,
    random_state,
    log_dir,
    cache_dir,
    n_jobs,
    model,
    pruning_strategy=None,
    filter_duplicated=False,
    objective_scaler="identity",
    max_steps=None,
    interval_steps=1,
    **kwargs,
):
    """Execute the CBO algorithm.

    Args:
        problem (HpProblem): problem (search space) definition.
        sync (bool): Boolean to execute the search in "synchronous" (``True``) or "asynchronous" (``False``) communication.
        strategy (str): strategy to use to generate batches of samples.
        timeout (int): duration in seconds of the search.
        max_evals (int): maximum number of evaluations for the search.
        random_state (int): random state/seed of the search.
        log_dir (str): path of the logging directory (i.e., where to store results).
        cache_dir (str): ...
    """
    rs = np.random.RandomState(random_state)
    rank_seed = rs.randint(low=0, high=2**32, size=size)[rank]

    hp_problem = problem.hp_problem
    run = problem.run

    # define where the outputs are saved live (in cache-dir if possible)
    if cache_dir is not None and os.path.exists(cache_dir):
        search_cache_dir = os.path.join(cache_dir, "search")
        search_log_dir = search_cache_dir
    else:
        search_log_dir = log_dir

    pathlib.Path(search_log_dir).mkdir(parents=False, exist_ok=True)

    if rank == 0:

        path_log_file = os.path.join(search_log_dir, "deephyper.log")
        logging.basicConfig(
            filename=path_log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
            force=True,
        )

        # Evaluator creation
        logging.info("Creation of the Evaluator...")

    if pruning_strategy:

        if pruning_strategy == "SHA":
            stopper = SuccessiveHalvingStopper(
                max_steps=max_steps,
                min_steps=1,
                min_fully_completed=1,
                reduction_factor=3, 
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


    storage = Storage.create(
        method="redis",
        method_kwargs={
            "host": os.environ["DEEPHYPER_DB_HOST"],
            "port": 6379,
        },
    )

    with Evaluator.create(
        run,
        method="mpicomm",
        method_kwargs={
            "storage": storage,
        },
    ) as evaluator:
        if evaluator is not None:
            logging.info(
                f"Creation of the Evaluator done with {evaluator.num_workers} worker(s)"
            )

            # Search
            logging.info("Creation of the search instance...")
            search = CBO(
                hp_problem,
                evaluator,
                sync_communication=sync,
                multi_points_strategy=strategy,
                n_jobs=n_jobs,
                log_dir=search_log_dir,
                random_state=rank_seed,
                acq_func=acq_func,
                surrogate_model=model,
                filter_failures="min",
                filter_duplicated=filter_duplicated,
                objective_scaler=objective_scaler,
                stopper=stopper,
            )
            logging.info("Creation of the search done")

            logging.info("Starting the search...")
            results = search.search(max_evals=max_evals, timeout=timeout)
            logging.info("Search is done")

            results.to_csv(os.path.join(search_log_dir, f"results.csv"))

            if log_dir != search_log_dir:  # means the cache was used
                os.system(f"mv {search_log_dir}/* {log_dir}")
