import warnings

warnings.simplefilter("ignore")

import argparse
import pathlib

import scalbo.benchmark.ackley
import scalbo.benchmark.hartmann6D
import scalbo.benchmark.frnn

import scalbo.search.ambs
import scalbo.search.dmbs


PROBLEMS = {
    "ackley": scalbo.benchmark.ackley,
    "hartmann6D": scalbo.benchmark.hartmann6D,
    "frnn": scalbo.benchmark.frnn,
}

SEARCHES = {
    "AMBS": scalbo.search.ambs,  # Centralized Model-Based Search (Master-Worker)
    "DMBS": scalbo.search.dmbs,  # Fully Distributed Model-Based Search
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
        "--sync",
        type=int,
        default=0,
        help="If the search workers must be syncronized or not.",
    )
    parser.add_argument(
        "--liar-strategy",
        type=str,
        default="boltzmann",
        choices=["cl_max", "topk", "boltzmann"],
        help="The liar strategy the optimizer must use.",
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
        "--acq-func",
        type=str,
        default="UCB",
        help="Acquisition funciton to use.",
    )

    return parser


def main(args):
    problem = PROBLEMS.get(args.problem)
    search = SEARCHES.get(args.search)
    sync = bool(args.sync)

    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    search.execute(
        problem,
        sync,
        args.liar_strategy,
        args.timeout,
        args.max_evals,
        args.random_state,
        args.log_dir,
        args.cache_dir,
        args.acq_func
    )


if __name__ == "__main__":
    parser = create_parser()

    args = parser.parse_args()

    main(args)
