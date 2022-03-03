import logging
import pathlib
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import time

import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"

from deephyper.evaluator import Evaluator
from deephyper.search.hps import AMBS
from deephyper.evaluator.callback import ProfilingCallback

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
        "ambs",
        "sync" if sync else "async",
        liar_strategy,
        str(timeout),
        "0" if max_evals == -1 else str(max_evals),
        str(random_state)
    ]
    
    if rank == 0:
        exp_name = (
            "-".join(config)
        )

        log_dir = os.path.join(log_dir_spec, exp_name)
        pathlib.Path(log_dir).mkdir(parents=False, exist_ok=True)

        log_file = os.path.join(log_dir, "deephyper.log")
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
        )

        # Evaluator creation
        logging.info("Creation of the Evaluator...")

    profiler = ProfilingCallback()

    with Evaluator.create(
        run,
        method="mpicomm",
        method_kwargs={
            "callbacks": [profiler],
        },
    ) as evaluator:
        if evaluator is not None:
            logging.info(
                f"Creation of the Evaluator done with {evaluator.num_workers} worker(s)"
            )

            # Search
            logging.info("Creation of the search instance...")
            search = AMBS(
                hp_problem,
                evaluator,
                log_dir=log_dir,
                liar_strategy=liar_strategy
            )
            logging.info("Creation of the search done")

            logging.info("Starting the search...")
            results = search.search(timeout=60 * timeout, max_evals=max_evals)
            logging.info("Search is done")

            results.to_csv(os.path.join(log_dir, "results.csv"))# f"{exp_name}.csv"))

            pathlib.Path("results").mkdir(parents=False, exist_ok=True)
            os.system(f"mv {log_dir} {os.path.join('results', exp_name)}")
            # os.system(f"rm results/{exp_name}/results.csv")
