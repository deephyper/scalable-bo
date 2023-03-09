import getpass
import logging
import pathlib
import os
import socket

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"

import numpy as np
import optuna
import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh

from deephyper.evaluator import SerialEvaluator, distributed
from deephyper.core.utils._timeout import terminate_on_timeout
from deephyper.core.exceptions import SearchTerminationError

from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init_thread()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def convert_to_optuna_distribution(cs_hp):
    name = cs_hp.name
    if isinstance(cs_hp, csh.UniformIntegerHyperparameter):
        if cs_hp.log:
            dist = optuna.distributions.IntLogUniformDistribution(
                low=cs_hp.lower, high=cs_hp.upper
            )
        else:
            dist = optuna.distributions.IntUniformDistribution(
                low=cs_hp.lower, high=cs_hp.upper
            )
    elif isinstance(cs_hp, csh.UniformFloatHyperparameter):
        if cs_hp.log:
            dist = optuna.distributions.LogUniformDistribution(
                low=cs_hp.lower, high=cs_hp.upper
            )
        else:
            dist = optuna.distributions.UniformDistribution(
                low=cs_hp.lower, high=cs_hp.upper
            )
    elif isinstance(cs_hp, csh.CategoricalHyperparameter):
        dist = optuna.distributions.CategoricalDistribution(choices=cs_hp.choices)
    elif isinstance(cs_hp, csh.OrdinalHyperparameter):
        dist = optuna.distributions.CategoricalDistribution(choices=cs_hp.sequence)
    else:
        raise TypeError(f"Cannot convert hyperparameter of type {type(cs_hp)}")

    return name, dist


def convert_to_optuna_space(cs_space):
    # verify pre-conditions
    if not (isinstance(cs_space, cs.ConfigurationSpace)):
        raise TypeError("Input space should be of type ConfigurationSpace")

    if len(cs_space.get_conditions()) > 0:
        raise RuntimeError("Cannot convert a ConfigSpace with Conditions!")

    if len(cs_space.get_forbiddens()) > 0:
        raise RuntimeError("Cannot convert a ConfigSpace with Forbiddens!")

    # convert the ConfigSpace to deephyper.skopt.space.Space
    distributions = {}
    for cs_hp in cs_space.get_hyperparameters():
        name, dist = convert_to_optuna_distribution(cs_hp)
        distributions[name] = dist

    return distributions


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

    optuna_space = convert_to_optuna_space(hp_problem.space)

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
            min_resource=1, reduction_factor=4
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

    if rank == 0:
        study = optuna.create_study(direction="maximize", **study_params)
    comm.Barrier()

    if rank > 0:
        study = optuna.load_study(**study_params)
    comm.Barrier()

    if (
        os.environ.get("HOST") and os.environ["HOST"][0] == "x"
    ):  # assume we are on Polaris
        backend = "s4m"
    else:
        backend = "mpi"
    evaluator = distributed(backend)(SerialEvaluator)(run)

    def execute_search():
        num_evals = 0
        while True:
            trial = study.ask(optuna_space)
            config = {p: trial.params[p] for p in trial.params}
            if pruning_strategy is not None:
                evaluator.run_function_kwargs["optuna_trial"] = trial

            evaluator.submit([config])
            local_results, other_results = evaluator.gather("ALL")
            num_evals += len(local_results) + len(other_results)
            y = local_results[0].objective
            step = local_results[0].metadata["budget"]

            evaluator.dump_evals(log_dir=log_dir)

            if max_evals > 0 and num_evals >= max_evals:
                break

            if isinstance(y, dict) and step is not None:  # pruner is used
                trial.report(y["objective"], step=step)
                if y["pruned"]:
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                else:
                    study.tell(trial, y)
            else:
                study.tell(trial, y)

    try:
        terminate_on_timeout(timeout, execute_search)
    except SearchTerminationError:
        logging.info("Search is done")
