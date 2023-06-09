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
import pandas as pd
import optuna
import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh

from deephyper.core.utils._timeout import terminate_on_timeout
from deephyper.evaluator import RunningJob

from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init_thread()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def optuna_suggest_from_hp(trial, cs_hp):
    name = cs_hp.name
    if isinstance(cs_hp, csh.UniformIntegerHyperparameter):
        value = trial.suggest_int(name, cs_hp.lower, cs_hp.upper, log=cs_hp.log)
    elif isinstance(cs_hp, csh.UniformFloatHyperparameter):
        value = trial.suggest_float(name, cs_hp.lower, cs_hp.upper, log=cs_hp.log)
    elif isinstance(cs_hp, csh.CategoricalHyperparameter):
        value = trial.suggest_categorical(name, cs_hp.choices)
        dist = optuna.distributions.CategoricalDistribution(choices=cs_hp.choices)
    elif isinstance(cs_hp, csh.OrdinalHyperparameter):
        value = trial.suggest_categorical(name, cs_hp.sequence)
    else:
        raise TypeError(f"Cannot convert hyperparameter of type {type(cs_hp)}")

    return name, value

def optuna_suggest_from_configspace(trial, cs_space):
    config = {}
    for cs_hp in cs_space.get_hyperparameters():
        name, value = optuna_suggest_from_hp(trial, cs_hp)
        config[name] = value
    return config

def execute_optuna(
    problem,
    timeout,
    max_evals,
    random_state,
    log_dir,
    cache_dir,
    method,  # in TPE
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
    logging.basicConfig(
        filename=path_log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
        force=True,
    )

    username = getpass.getuser()
    host = os.environ["OPTUNA_DB_HOST"]

    storage = f"postgresql://{username}@{host}:5432/hpo"

    logging.info(f"storage={storage}")

    if "TPE" in method:
        sampler = optuna.samplers.TPESampler(seed=rank_seed)
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
    study_params = dict(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
    )

    timestamp_start = None
    if rank == 0:
        timestamp_start = time.time()
        study = optuna.create_study(direction="maximize", **study_params)
    comm.Barrier()
    timestamp_start = comm.bcast(timestamp_start)

    if rank > 0:
        study = optuna.load_study(**study_params)
    comm.Barrier()

    def objective_wrapper(trial):
        config = optuna_suggest_from_configspace(trial, hp_problem.space)
        output = run(RunningJob(id=trial.number, parameters=config), optuna_trial=trial)

        data = {f"p:{k}": v for k, v in config.items()}
        data["objective"] = output["objective"]
        data["job_id"] = trial.number
        data.update({f"m:{k}": v for k, v in output["metadata"].items()})
        trial.set_user_attr("results", data)

        if data["m:stopped"]:
            raise optuna.TrialPruned()
        
        return data["objective"]
    
    def optimize_wrapper(duration):
        study.optimize(objective_wrapper, timeout=duration)

    timestamp_start = time.time()
    optimize = functools.partial(terminate_on_timeout, timeout, optimize_wrapper)
    optimize(timeout)

    all_trials = study.get_trials(deep=True, states=[optuna.trial.TrialState.COMPLETE])

    pd.DataFrame([t.user_attrs["results"] for t in all_trials]).to_csv(os.path.join(search_log_dir, "results.csv"))

    logging.info("Search is done")
