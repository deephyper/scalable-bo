import functools
import getpass
import logging
import pathlib
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"

import numpy as np
import optuna

from deephyper.search.hps._mpi_doptuna import MPIDistributedOptuna

from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init_thread()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def execute_optuna(
    problem,
    timeout,
    max_evals,
    random_state,
    log_dir,
    cache_dir,
    method,
    pruning_strategy=None,  # SHA, HB
    max_steps=None,
    **kwargs,
):
    """Execute the HB algorithm.

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

    path_log_file = os.path.join(search_log_dir, f"deephyper.{rank}.log")
    if rank == 0:
        logging.basicConfig(
            filename=path_log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
            force=True,
        )

    username = getpass.getuser()
    host = os.environ["OPTUNA_DB_HOST"]
    storage = f"postgresql://{username}@{host}:5432/hpo"
    n_objectives = int(os.environ.get("OPTUNA_N_OBJECTIVES", 1))

    logging.info(f"storage={storage}")

    if "TPE" in method:
        sampler = optuna.samplers.TPESampler(seed=rank_seed)
    elif "NSGAII" in method:
        sampler = optuna.samplers.NSGAIISampler(seed=rank_seed)
    else:
        sampler = optuna.samplers.RandomSampler(seed=rank_seed)

    if pruning_strategy is None or pruning_strategy == "NONE":
        pruner = optuna.pruners.NopPruner()
    elif pruning_strategy == "SHA":
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1, reduction_factor=3
        )
    elif pruning_strategy == "HB":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=max_steps,  #! careful!
            reduction_factor=4,
        )
    elif pruning_strategy == "MED":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=30, interval_steps=10
        )
    else:
        raise ValueError(f"Wrong pruning strategy '{pruning_strategy}'")

    study_name = os.path.basename(log_dir)

    search = MPIDistributedOptuna(
        hp_problem,
        run,
        random_state=random_state,
        log_dir=search_log_dir,
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        comm=comm,
        n_objectives=n_objectives,
        verbose=0,
    )
    results = search.search(max_evals=max_evals, timeout=timeout)

    logging.info("Search is done")
    
    MPI.Finalize()
