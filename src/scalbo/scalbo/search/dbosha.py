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

import optuna

import numpy as np

from deephyper.search.hps import DBO
from deephyper.evaluator import distributed
from ._pruning_evaluator import PruningEvaluator

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
    n_jobs,
    model,
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

    # Optuna Pruner
    username = getpass.getuser()
    host = os.environ["OPTUNA_DB_HOST"]

    storage = f"postgresql://{username}@{host}:5432/hpo"

    study_name = os.path.basename(log_dir)
    study_params = dict(
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.RandomSampler(random_state),
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3),
    )

    if rank == 0:
        study = optuna.create_study(direction="maximize", **study_params)
    comm.Barrier()

    if rank > 0:
        study = optuna.load_study(**study_params)
    comm.Barrier()

    evaluator = distributed(backend="mpi")(PruningEvaluator)(run, study=study)

    logging.info("Creation of the search instance...")

    search = DBO(
        hp_problem,
        evaluator,
        sync_communication=sync,
        n_jobs=n_jobs,
        log_dir=search_log_dir,
        random_state=rs,
        acq_func=acq_func,
        surrogate_model=model,
        filter_duplicated=False,
    )
    logging.info("Creation of the search done")

    results = None
    logging.info("Starting the search...")
    if rank == 0:
        results = search.search(timeout=timeout)
    else:
        search.search(timeout=timeout)
    logging.info("Search is done")

    if log_dir != search_log_dir:

        if rank == 0:
            os.system(f"mv {search_log_dir}/results.csv {log_dir}")

            os.system(f"mv {path_log_file} {log_dir}")
