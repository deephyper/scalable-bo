import warnings

warnings.simplefilter("ignore")

import argparse
import pathlib
import importlib
import sys

PROBLEMS = {
    "fast_ackley_2": "scalbo.benchmark.fast_ackley_2",
    "fastest_ackley_2": "scalbo.benchmark.fastest_ackley_2",
    "hb_sim": "scalbo.benchmark.hb_sim",
    "dackley_2": "scalbo.benchmark.dackley_2",
    "dackley_5": "scalbo.benchmark.dackley_5",
    "ackley_5": "scalbo.benchmark.ackley_5",
    "ackley_10": "scalbo.benchmark.ackley_10",
    "ackley_30": "scalbo.benchmark.ackley_30",
    "ackley_50": "scalbo.benchmark.ackley_50",
    "ackley_100": "scalbo.benchmark.ackley_100",
    "hartmann6D": "scalbo.benchmark.hartmann6D",
    "levy": "scalbo.benchmark.levy",
    "griewank": "scalbo.benchmark.griewank",
    "schwefel": "scalbo.benchmark.schwefel",
    "frnn": "scalbo.benchmark.frnn",
    "minimalistic-frnn": "scalbo.benchmark.minimalistic_frnn",
    "molecular": "scalbo.benchmark.molecularmpnn",
    "candle_attn": "scalbo.benchmark.candle_attn",
    "candle_attn_sim": "scalbo.benchmark.candle_attn_sim",
    "candle_combo": "scalbo.benchmark.candle_combo",
    "dhb_combo": "scalbo.benchmark.dhb_combo",
    "test": "scalbo.benchmark.test",
}

SEARCHES = {
    "CBO": "scalbo.search.cbo",  # Centralized Model-Based Search (Master-Worker)
    "DBO": "scalbo.search.dbo",  # Fully Distributed Model-Based Search
    "DBOS4M": "scalbo.search.dbos4m",  # Fully Distributed Model-Based Search
    "TPEHB": "scalbo.search.tpehb",  # TPEHB
    "TPESHA": "scalbo.search.tpesha", # TPESHA
    "DBOSHA": "scalbo.search.dbosha", # DBOSHA
    "DBOS4MSHA": "scalbo.search.dbos4msha", # DBOSHA
}


def create_parser():
    parser = argparse.ArgumentParser(description="Command line to run experiments.")

    parser.add_argument(
        "--problem",
        type=str,
        choices=list(PROBLEMS.keys()),
        required=True,
        help="Problem on which to experiment.",
    )
    parser.add_argument(
        "--search",
        type=str,
        choices=list(SEARCHES.keys()),
        required=True,
        help="Search the experiment must be done with.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["RF", "GP", "DUMMY", "MF"],
        required=False,
        default="RF",
        help="Surrogate model used by the Bayesian optimizer.",
    )
    parser.add_argument(
        "--sync",
        type=int,
        default=0,
        help="If the search workers must be syncronized or not.",
    )
    parser.add_argument(
        "--acq-func",
        type=str,
        default="UCB",
        help="Acquisition funciton to use.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="boltzmann",
        choices=["cl_max", "topk", "boltzmann", "qUCB", "qEI"],
        help="The strategy for multi-point acquisition.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Search maximum duration (in min.) for each optimization.",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=-1,
        help="Number of iterations to run for each optimization.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Control the random-state of the algorithm.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="output",
        help="Logging directory to store produced outputs.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Path to use to cache logged outputs (e.g., /dev/shm/).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=bool,
        default=False,
        help="Wether to activate or not the verbose mode.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="The number of parallel processes to use to fit the surrogate model.",
    )

    return parser


def main(args):

    # load the problem
    problem = importlib.import_module(PROBLEMS.get(args.problem))
    search = importlib.import_module(SEARCHES.get(args.search))
    sync = bool(args.sync)

    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    search.execute(
        problem,
        sync,
        args.acq_func,
        args.strategy,
        args.timeout,
        args.random_state,
        args.log_dir,
        args.cache_dir,
        args.n_jobs,
        args.model,
    )


if __name__ == "__main__":
    parser = create_parser()

    args = parser.parse_args()

    # delete arguments to avoid conflicts
    sys.argv = [sys.argv[0]]

    main(args)
