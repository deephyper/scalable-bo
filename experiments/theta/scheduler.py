import warnings

warnings.simplefilter("ignore")

import argparse
import pathlib

import os

import yaml

try:
    from yaml import CLoader as Loader
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import scalbo.benchmark.ackley

import scalbo.search.ambs
import scalbo.search.dmbs

PROBLEMS = {"ackley": scalbo.benchmark.ackley}

SEARCHES = {
    "ambs": scalbo.search.ambs,  # Centralized Model-Based Search (Master-Worker)
    "dmbs": scalbo.search.dmbs,  # Fully Distributed Model-Based Search
}

import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"

from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init_thread()
    
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def create_parser():
    parser = argparse.ArgumentParser(description="Command line to run experiments.")

    parser.add_argument(
        "--configs",
        type=str,
        required=True,
        help="The .yaml file in which are specified configurations tu run.",
    )

    parser.add_argument(
        "--num-nodes",
        type=int,
        required=True,
        help="The number of nodes on which is run the exp.",
    )

    parser.add_argument(
        "--ranks-per-node",
        type=int,
        required=True,
        help="The number of ranks per node on which is run the exp.",
    )

    return parser


def run_config(problem, search, sync, liar_strategy, timeout, max_evals, random_state, log_dir, cache_dir):
    problem = PROBLEMS.get(problem)
    search = SEARCHES.get(search)
    sync = sync == 'sync'

    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    search.execute(
        problem,
        sync,
        liar_strategy,
        timeout,
        max_evals,
        random_state,
        log_dir,
        cache_dir
    )

    comm.Barrier()


def main(args):

    num_nodes = args.num_nodes
    ranks_per_node = args.ranks_per_node

    with open(args.configs, "r") as f:
        configs = yaml.load(f, Loader=Loader)
    
    problems = configs["problems"]
    searches = configs["searches"]
    syncs = configs["syncs"]
    liar_strategies = configs["liar-strategies"]
    timeouts = configs["timeouts"]
    max_evals = configs["max-evals"]
    random_states = configs["random-states"]

    log_dir = configs["log-dir"]
    cache_dir = configs["cache-dir"]

    for problem in problems:
        for search in searches:
            for sync in syncs:
                for liar_strategy in liar_strategies:
                    for timeout in timeouts:
                        for max_eval in max_evals:
                            for random_state in random_states:
                                exp = f"{problem}-{search}-{sync}-{liar_strategy}-{num_nodes}-{ranks_per_node}-{timeout}-{random_state}"
                                if rank==0:
                                    print(f"Running: {exp}")
                                log = os.path.join(log_dir, exp)
                                try:
                                    run_config(problem, search, sync, liar_strategy, timeout, max_eval, random_state, log, cache_dir)
                                except Exception as e:
                                    if rank==0:
                                        print(f"An issue occured with {exp} : {e}")

if __name__ == "__main__":

    parser = create_parser()

    args = parser.parse_args()

    main(args)
